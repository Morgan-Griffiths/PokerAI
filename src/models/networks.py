
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from models.utils import mask_

from models.buffers import PriorityReplayBuffer,PriorityTree
import hand_recognition.datatypes as dt

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

def hard_update(source,target):
    for target_param,param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def norm_frequencies(action_soft,mask):
    # with torch.no_grad():
    action_masked = action_soft * mask
    action_probs =  action_masked / action_masked.sum(-1).unsqueeze(-1)
    return action_probs

def combined_masks(action_mask,betsize_mask):
    """Combines action and betsize masks into flat mask for 1d network outputs"""
    if action_mask.dim() > 1:
        return torch.cat([action_mask[:,:-2],betsize_mask],dim=-1)
    else:
        return torch.cat([action_mask[:-2],betsize_mask])

class NetworkFunctions(object):
    def __init__(self,nA,nB):
        self.nA = nA
        self.nB = nB
        self.nC = nA - 2 + self.nB

    def wrap_action(self,action,betsize_category,previous_action):
        """
        Wraps split action/betsize into flat action.
        Bets and raises are combined into one.
        """
        actions = torch.zeros(self.nC)
        if action < 3:
            actions[action] = 1
        else: # Bet or raise
            actions[betsize_category + 3] = 1
        return torch.argmax(actions, dim=0).unsqueeze(0)

    def unwrap_action(self,action:torch.Tensor,previous_action:torch.Tensor):
        """Unwraps flat action into action_category and betsize_category"""
        # print(action,previous_action)
        actions = torch.zeros(self.nA)
        betsizes = torch.zeros(self.nB)
        # actions[action[action < 3]] = 1
        if action < 3:
            actions[action] = 1
        elif previous_action == 5 or previous_action == 0: # Unopened
            actions[3] = 1
            bet_category = action - 3
            betsizes[bet_category] = 1
        else: # facing bet or raise
            actions[4] = 1
            bet_category = action - 3
            betsizes[bet_category] = 1
        int_actions = torch.argmax(actions, dim=0).unsqueeze(0)
        int_betsizes = torch.argmax(betsizes, dim=0).unsqueeze(0)
        return int_actions,int_betsizes

################################################
#                Helper Layers                 #
################################################

class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0).float()#.to(device)

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x 

class Embedder(nn.Module):
    def __init__(self,vocab_size,d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size,d_model)
    def forward(self,x):
        return self.embed(x)

class SelfAttentionWide(nn.Module):
    def __init__(self, emb, heads=8, mask=False):
        """
        :param emb:
        :param heads:
        :param mask:
        """

        super().__init__()

        self.emb = emb
        self.heads = heads
        self.mask = mask

        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)

        self.unifyheads = nn.Linear(heads * emb, emb)

    def forward(self, x):

        b, t, e = x.size()
        h = self.heads
        assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'

        keys    = self.tokeys(x)   .view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values  = self.tovalues(x) .view(b, t, h, e)

        # compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        queries = queries / (e ** (1/4))
        keys    = keys / (e ** (1/4))
        # - Instead of dividing the dot products by sqrt(e), we scale the keys and values.
        #   This should be more memory efficient

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))

        assert dot.size() == (b*h, t, t)

        if self.mask: # mask out the upper half of the dot matrix, excluding the diagonal
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        dot = F.softmax(dot, dim=2)
        # - dot now has row-wise self-attention probabilities

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, e)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)

        return self.unifyheads(out)

class SelfAttentionNarrow(nn.Module):

    def __init__(self, emb, heads=8, mask=False):
        """
        :param emb:
        :param heads:
        :param mask:
        """

        super().__init__()

        assert emb % heads == 0, f'Embedding dimension ({emb}) should be divisible by nr. of heads ({heads})'

        self.emb = emb
        self.heads = heads
        self.mask = mask

        s = emb // heads
        # - We will break the embedding into `heads` chunks and feed each to a different attention head

        self.tokeys    = nn.Linear(s, s, bias=False)
        self.toqueries = nn.Linear(s, s, bias=False)
        self.tovalues  = nn.Linear(s, s, bias=False)

        self.unifyheads = nn.Linear(heads * s, emb)

    def forward(self, x):

        b, t, e = x.size()
        h = self.heads
        assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'

        s = e // h
        x = x.view(b, t, h, s)

        keys    = self.tokeys(x)
        queries = self.toqueries(x)
        values  = self.tovalues(x)

        assert keys.size() == (b, t, h, s)
        assert queries.size() == (b, t, h, s)
        assert values.size() == (b, t, h, s)

        # Compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
        values = values.transpose(1, 2).contiguous().view(b * h, t, s)

        queries = queries / (e ** (1/4))
        keys    = keys / (e ** (1/4))
        # - Instead of dividing the dot products by sqrt(e), we scale the keys and values.
        #   This should be more memory efficient

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))

        assert dot.size() == (b*h, t, t)

        if self.mask: # mask out the upper half of the dot matrix, excluding the diagonal
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        dot = F.softmax(dot, dim=2)
        # - dot now has row-wise self-attention probabilities

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, s)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, s * h)

        return self.unifyheads(out)

class TransformerBlock(nn.Module):

    def __init__(self, emb, heads, mask, seq_length, ff_hidden_mult=4, dropout=0.0, wide=True):
        super().__init__()

        self.attention = SelfAttentionWide(emb, heads=heads, mask=mask) if wide \
                    else SelfAttentionNarrow(emb, heads=heads, mask=mask)
        self.mask = mask

        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )

        self.do = nn.Dropout(dropout)

    def forward(self, x):

        attended = self.attention(x)

        x = self.norm1(attended + x)

        x = self.do(x)

        fedforward = self.ff(x)

        x = self.norm2(fedforward + x)

        x = self.do(x)

        return x

class CTransformer(nn.Module):
    """
    Transformer for classifying sequences
    """

    def __init__(self, emb, heads, depth, seq_length, num_classes, max_pool=True, dropout=0.0, wide=False):
        """
        :param emb: Embedding dimension
        :param heads: nr. of attention heads
        :param depth: Number of transformer blocks
        :param seq_length: Expected maximum sequence length
        :param num_tokens: Number of tokens (usually words) in the vocabulary
        :param num_classes: Number of classes.
        :param max_pool: If true, use global max pooling in the last layer. If false, use global
                         average pooling.
        """
        super().__init__()

        self.max_pool = max_pool

        # self.token_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=num_tokens)
        # self.pos_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=seq_length)

        tblocks = []
        for i in range(depth):
            tblocks.append(
                TransformerBlock(emb=emb, heads=heads, seq_length=seq_length, mask=False, dropout=dropout, wide=wide))

        self.tblocks = nn.Sequential(*tblocks)

        self.toprobs = nn.Linear(emb, num_classes)

        self.do = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x: A batch by sequence length integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        # tokens = self.token_embedding(x)
        # b, t, e = tokens.size()

        # positions = self.pos_embedding(torch.arange(t, device=d()))[None, :, :].expand(b, t, e)
        # x = tokens + positions
        x = self.do(x)

        x = self.tblocks(x)

        x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1) # pool over the time dimension

        x = self.toprobs(x)

        return F.log_softmax(x, dim=1)


class ProcessHandBoard(nn.Module):
    def __init__(self,params,critic=False):
        super().__init__()
        self.one_hot_suits = torch.nn.functional.one_hot(torch.arange(0,dt.SUITS.HIGH))
        self.one_hot_ranks = torch.nn.functional.one_hot(torch.arange(0,dt.RANKS.HIGH))
        # Input is (b,4,2) -> (b,4,4) and (b,4,13)
        self.suit_conv = nn.Sequential(
            nn.Conv1d(7, 16, kernel_size=1, stride=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )
        self.rank_conv = nn.Sequential(
            nn.Conv1d(7, 16, kernel_size=5, stride=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )
        self.max_length = params['max_length']
        self.initialize(critic)

    def initialize(self,critic):
        if critic:
            self.forward = self.forward_critic
        else:
            self.forward = self.forward_actor

    def forward_critic(self,x):
        """x: concatenated hand and board. alternating rank and suit."""
        B,M,C = x.size()
        ranks = x[:,:,::2]
        suits = x[:,:,1::2]
        hero_ranks = ranks[:,:,:2]
        villain_ranks = ranks[:,:,2:4]
        board_ranks = ranks[:,:,4:]
        hero_suits = suits[:,:,:2]
        villain_suits = suits[:,:,2:4]
        board_suits = suits[:,:,4:]
        hero_hand_ranks = torch.cat((hero_ranks,board_ranks),dim=-1)
        hero_hand_suits = torch.cat((hero_suits,board_suits),dim=-1)
        villain_hand_ranks = torch.cat((villain_ranks,board_ranks),dim=-1)
        villain_hand_suits = torch.cat((villain_suits,board_suits),dim=-1)
        hero_hot_ranks = self.one_hot_ranks[hero_hand_ranks]
        hero_hot_suits = self.one_hot_suits[hero_hand_suits]
        villain_hot_ranks = self.one_hot_ranks[villain_hand_ranks]
        villain_hot_suits = self.one_hot_suits[villain_hand_suits]
        hero_activations = []
        villain_activations = []
        for i in range(M):
            hero_s = self.suit_conv(hero_hot_suits[:,i,:,:].float())
            hero_r = self.rank_conv(hero_hot_ranks[:,i,:,:].float())
            hero_activations.append(torch.cat((hero_r,hero_s),dim=-1))
            villain_s = self.suit_conv(villain_hot_suits[:,i,:,:].float())
            villain_r = self.rank_conv(villain_hot_ranks[:,i,:,:].float())
            villain_activations.append(torch.cat((villain_r,villain_s),dim=-1))
        hero = torch.stack(hero_activations).view(B,M,-1)
        villain = torch.stack(villain_activations).view(B,M,-1)
        return hero - villain

    def forward_actor(self,x):
        """x: concatenated hand and board. alternating rank and suit."""
        B,M,C = x.size()
        ranks = x[:,:,::2]
        suits = x[:,:,1::2]
        hot_ranks = self.one_hot_ranks[ranks]
        hot_suits = self.one_hot_suits[suits]
        activations = []
        for i in range(M):
            s = self.suit_conv(hot_suits[:,i,:].float())
            r = self.rank_conv(hot_ranks[:,i,:].float())
            activations.append(torch.cat((r,s),dim=-1))
        return torch.stack(activations).view(B,M,-1)

class ProcessOrdinal(nn.Module):
    def __init__(self,params):
        super().__init__()
        self.mapping = params['mapping']
        self.street_emb = nn.Embedding(embedding_dim=params['embedding_size'], num_embeddings=4)
        self.action_emb = nn.Embedding(embedding_dim=params['embedding_size'], num_embeddings=6)
        self.position_emb = nn.Embedding(embedding_dim=params['embedding_size'], num_embeddings=2)
        self.order_emb = nn.Embedding(embedding_dim=params['embedding_size'], num_embeddings=2)

    def forward(self,x):
        order = self.order_emb(torch.arange(2))
        street = self.street_emb(x[:,:,0].long())
        hero_position = self.position_emb(x[:,:,1].long()) + order[0]
        vil_position = self.position_emb(x[:,:,2].long()) + order[1]
        previous_action = self.action_emb(x[:,:,3].long())
        ordinal_output = torch.cat((street,hero_position,vil_position,previous_action),dim=-1)
        return ordinal_output

class ProcessContinuous(nn.Module):
    def __init__(self,params):
        super().__init__()
        self.mapping = params['mapping']
        self.betsize_fc = nn.Linear(1,params['embedding_size'])
        self.stack_fc = nn.Linear(1,params['embedding_size'])
        self.call_fc = nn.Linear(1,params['embedding_size'])
        self.odds_fc = nn.Linear(1,params['embedding_size'])
        self.order_emb = nn.Embedding(embedding_dim=params['embedding_size'], num_embeddings=5)
        
    def forward(self,x):
        B,M,C = x.size()
        # 
        previous_betsize = x[:,:,0]
        hero_stack = x[:,:,1]
        villain_stack = x[:,:,2]
        amnt_to_call = x[:,:,3]
        pot_odds = x[:,:,4]

        order = self.order_emb(torch.arange(5))
        bets = []
        heros = []
        villains = []
        calls = []
        odds = []
        for i in range(M):
            bets.append(self.betsize_fc(previous_betsize[:,i]) + order[0])
            heros.append(self.stack_fc(hero_stack[:,i]) + order[1])
            villains.append(self.stack_fc(villain_stack[:,i]) + order[2])
            calls.append(self.call_fc(amnt_to_call[:,i]) + order[3])
            odds.append(self.odds_fc(pot_odds[:,i]) + order[4])
        bet = torch.stack(bets)
        hero = torch.stack(heros)
        villain = torch.stack(villains)
        call = torch.stack(calls)
        odd = torch.stack(odds)
        continuous_output = torch.stack((bet,hero,villain,call,odd),dim=-1).view(B,M,-1)
        return continuous_output

class PreProcessPokerInputs(nn.Module):
    def __init__(self,params,critic=False):
        super().__init__()
        self.mapping = params['mapping']
        self.hand_board = ProcessHandBoard(params,critic)
        self.continuous = ProcessContinuous(params)
        self.ordinal = ProcessOrdinal(params)
        self.initialize(critic)

    def initialize(self,critic):
        if critic:
            self.forward = self.forward_critic
        else:
            self.forward = self.forward_actor

    def forward_critic(self,x):
        h = self.hand_board(x[:,:,self.mapping['observation']['hand_board']].long())
        # h.size(B,M,240)
        o = self.continuous(x[:,:,self.mapping['observation']['continuous'].long()])
        # o.size(B,M,5)
        c = self.ordinal(x[:,:,self.mapping['observation']['ordinal'].long()])
        # h.size(B,M,128)
        combined = torch.cat((h,o,c),dim=-1)
        return combined

    def forward_actor(self,x):
        h = self.hand_board(x[:,:,self.mapping['state']['hand_board']].long())
        # h.size(B,M,240)
        o = self.continuous(x[:,:,self.mapping['state']['continuous'].long()])
        # o.size(B,M,5)
        c = self.ordinal(x[:,:,self.mapping['state']['ordinal'].long()])
        # h.size(B,M,128)
        combined = torch.cat((h,o,c),dim=-1)
        return combined

################################################
#               Holdem Networks                #
################################################

class HoldemBaseline(nn.Module):
    def __init__(self,seed,nS,nA,nB,params,hidden_dims=(64,64),activation=F.leaky_relu):
        super().__init__()
        self.activation = activation
        self.nS = nS
        self.nA = nA
        self.nB = nB
        self.combined_output = nA - 2 + nB
        self.helper_functions = NetworkFunctions(self.nA,self.nB)
        self.max_length = params['max_length']
        self.process_input = PreProcessPokerInputs(params)
        
        self.seed = torch.manual_seed(seed)
        self.mapping = params['mapping']
        self.hand_emb = Embedder(5,64)
        self.action_emb = Embedder(6,64)
        self.betsize_emb = Embedder(self.nB,64)
        self.noise = GaussianNoise()
        emb = 528
        n_heads = 8
        depth = 2
        self.transformer = CTransformer(emb,n_heads,depth,self.max_length,self.nA)

        self.fc1 = nn.Linear(528,hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0],hidden_dims[1])
        self.fc3 = nn.Linear(64,self.combined_output)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self,state,action_mask,betsize_mask):
        mask = combined_masks(action_mask,betsize_mask)
        x = state
        B,M,c = x.size()
        n_padding = self.max_length - M
        out = self.process_input(x)
        padding = torch.zeros(B,n_padding,out.size(-1))
        h = torch.cat((out,padding),dim=1)
        # should be (b,64,88)
        action_logits = self.transformer(h)
        category_logits = self.noise(action_logits)
        
        action_soft = F.softmax(category_logits,dim=-1)
        action_probs = norm_frequencies(action_soft,mask)
        m = Categorical(action_probs)
        action = m.sample()

        action_category,betsize_category = self.helper_functions.unwrap_action(action,state[:,-1,self.mapping['state']['previous_action']])
        # print('state',state)
        # print('action_category,betsize_category',action_category,betsize_category)
        
        outputs = {
            'action':action,
            'action_category':action_category,
            'action_prob':m.log_prob(action),
            'action_probs':action_probs,
            'betsize':betsize_category
            }
        return outputs

class HoldemBaselineCritic(nn.Module):
    def __init__(self,seed,nO,nA,nB,params,hidden_dims=(64,64),activation=F.leaky_relu):
        super().__init__()
        self.activation = activation
        self.nO = nO
        self.nA = nA
        
        self.seed = torch.manual_seed(seed)
        self.mapping = params['mapping']

        self.process_input = PreProcessPokerInputs(params,critic=True)
        self.fc1 = nn.Linear(304,hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0],hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1],nA)
        self.dropout = nn.Dropout(0.5)
        self.value_output = nn.Linear(64,1)
        self.advantage_output = nn.Linear(64,self.nA)

    def forward(self,x,action):
        M,c = x.size()
        ranks = x[:,self.mapping['observation']['rank']].long()
        suits = x[:,self.mapping['observation']['suit']].long()
        vil_rank = x[:,self.mapping['observation']['vil_ranks']].long()
        vil_suit = x[:,self.mapping['observation']['vil_suits']].long()
        board_ranks = x[:,self.mapping['observation']['board_ranks']].long()
        board_suits = x[:,self.mapping['observation']['board_suits']].long()

        rank_input = torch.cat((ranks,board_ranks),dim=-1)
        suit_input = torch.cat((suits,board_suits),dim=-1)
        hot_ranks = self.one_hot_ranks[rank_input]
        hot_suits = self.one_hot_suits[suit_input]

        s = self.suit_conv(hot_suits.float())
        r = self.rank_conv(hot_ranks.float())
        hero = torch.cat((r,s),dim=-1)

        rank_input2 = torch.cat((vil_rank,board_ranks),dim=-1)
        suit_input2 = torch.cat((vil_suit,board_suits),dim=-1)
        hot_ranks2 = self.one_hot_ranks[rank_input2]
        hot_suits2 = self.one_hot_suits[suit_input2]

        s2 = self.suit_conv(hot_suits2.float())
        r2 = self.rank_conv(hot_ranks2.float())
        villain = torch.cat((r2,s2),dim=-1)
        # should be (b,64,88)

        winner = hero - villain

        last_action = x[:,self.mapping['observation']['previous_action']].long()
        last_action = self.action_emb(last_action)
        x = torch.cat([winner.view(M,-1),last_action],dim=-1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        outputs = {
            'value':torch.tanh(self.fc3(x))
            }
        return outputs

class HoldemQCritic(nn.Module):
    def __init__(self,seed,nO,nA,nB,params,hidden_dims=(64,64),activation=F.leaky_relu):
        super().__init__()
        self.activation = activation
        self.nO = nO
        self.nA = nA
        
        self.process_input = PreProcessPokerInputs(params,critic=True)
        self.seed = torch.manual_seed(seed)
        self.max_length = params['max_length']
        self.mapping = params['mapping']
        self.fc1 = nn.Linear(528,hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0],hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1],nA)
        emb = 528
        n_heads = 8
        depth = 2
        self.transformer = CTransformer(emb,n_heads,depth,self.max_length,self.nA)
        self.dropout = nn.Dropout(0.5)
        self.value_output = nn.Linear(5,1)
        self.advantage_output = nn.Linear(5,self.nA)

    def forward(self,obs):
        x = obs
        B,M,c = x.size()
        out = self.process_input(x)
        transformer_outs = []
        for i in range(M):
            transformer_input = out[:,i,:].unsqueeze(1)
            transformer_outs.append(self.transformer(transformer_input))
        h = torch.stack(transformer_outs).permute(1,0,2)
        # n_padding = self.max_length - M
        # padding = torch.zeros(B,n_padding,out.size(-1))
        # h = torch.cat((out,padding),dim=1)
        # x = self.activation(self.fc1(out))
        # x = self.activation(self.fc2(x))
        # x = self.activation(self.fc3(x))
        # x = self.dropout(x)
        a = self.advantage_output(h)
        v = self.value_output(h)
        v = v.expand_as(a)
        q = v + a - a.mean(-1,keepdim=True).expand_as(a)
        outputs = {
            'value':q.squeeze(0)
            }
        return outputs

################################################
#                Kuhn Networks                 #
################################################

################################################
#              Betsize Networks                #
################################################

class BetsizeActor(nn.Module):
    def __init__(self,seed,nS,nC,nA,params,hidden_dims=(64,64),activation=F.leaky_relu):
        """
        Num Categories: nC (check,fold,call,bet,raise)
        Num Betsizes: nA (various betsizes)
        """
        super().__init__()
        self.activation = activation
        self.nS = nS
        self.nC = nC
        self.nA = nA
        
        self.seed = torch.manual_seed(seed)
        self.mapping = params['mapping']
        self.hand_emb = Embedder(5,64)
        self.action_emb = Embedder(6,64)
        self.betsize_emb = Embedder(self.nA,64)
        self.noise = GaussianNoise()
        self.fc1 = nn.Linear(128,hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0],hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1],nC)
        self.bfc1 = nn.Linear(64,hidden_dims[0])
        self.bfc2 = nn.Linear(hidden_dims[0],hidden_dims[1])
        self.bfc3 = nn.Linear(hidden_dims[1],nA)
        
    def forward(self,state,mask,betsize_mask):
        x = state
        M,c = x.size()
        hand = x[:,self.mapping['state']['rank']].long()
        last_action = x[:,self.mapping['state']['previous_action']].long()
        # previous_betsize = x[:,self.mapping['state']['previous_betsize']].float().unsqueeze(0)
        hand = self.hand_emb(hand)
        embedded_action = self.action_emb(last_action)
        # print(hand.size(),embedded_action.size(),previous_betsize.size())
        # x = torch.cat([hand,embedded_action,previous_betsize],dim=-1)
        x = torch.cat([hand,embedded_action],dim=-1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        category_logits = self.fc3(x)
        category_logits = self.noise(category_logits)
        action_soft = F.softmax(category_logits,dim=-1)
        action_probs = norm_frequencies(action_soft,mask)
        # with torch.no_grad():
        #     action_masked = action_soft * mask
        #     action_probs =  action_masked / action_masked.sum(-1).unsqueeze(1)
        m = Categorical(action_probs)
        action = m.sample()
        # Check which category it is
        # betsize = torch.tensor([-1])
        # betsize_prob = torch.tensor([-1]).float()
        # betsize_probs = torch.Tensor(self.nA).fill_(-1).unsqueeze(0).float()
        # # print('action',action)
        # # print('betsize_mask',betsize_mask)
        # if action > 2:
        # generate betsize
        b = self.activation(self.bfc1(x))
        b = self.activation(self.bfc2(b))
        b = self.bfc3(b)
        betsize_logits = self.noise(b)
        # print('betsize_logits',betsize_logits)
        betsize_probs = F.softmax(betsize_logits,dim=-1)
        # print('betsize_probs',betsize_probs)
        if betsize_mask.sum(-1) == 0:
            betsize_mask = torch.ones(M,self.nA)
        # with torch.no_grad():
        mask_betsize_probs = betsize_probs * betsize_mask
        # print('mask_betsize_probs',mask_betsize_probs)
        norm_betsize_probs = mask_betsize_probs / mask_betsize_probs.sum(-1).unsqueeze(1)
        # print('mask_betsize_probs',mask_betsize_probs)
        b = Categorical(norm_betsize_probs)
        betsize = b.sample()
        betsize_prob = b.log_prob(betsize)

        # print('betsize',betsize)
        # print('betsize_prob',betsize_prob)
        # print('betsize_probs',betsize_probs)
        outputs = {
            'action':action,
            'action_prob':m.log_prob(action),
            'action_probs':action_probs,
            'action_category':action,
            'betsize':betsize,
            'betsize_prob':betsize_prob,
            'betsize_probs':betsize_probs}
        return outputs

class BetsizeCritic(nn.Module):
    def __init__(self,seed,nS,nC,nA,params,hidden_dims=(64,64),activation=F.leaky_relu):
        super().__init__()
        self.activation = activation
        self.nS = nS
        self.nC = nC
        self.nA = nA
        
        self.seed = torch.manual_seed(seed)
        self.use_embedding = params['embedding']
        self.mapping = params['mapping']
        self.one_hot_kuhn = torch.nn.functional.one_hot(torch.arange(0,4))
        self.one_hot_actions = torch.nn.functional.one_hot(torch.arange(0,6))
        self.hand_emb = Embedder(5,32)
        self.action_emb = Embedder(6,32)
        self.positional_embeddings = Embedder(2,32)

        self.conv = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=3, stride=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )
        self.fc0 = nn.Linear(64,hidden_dims[0])
        self.fc1 = nn.Linear(97,hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0],hidden_dims[1])
        self.value_output = nn.Linear(64,1)
        self.advantage_output = nn.Linear(64,self.nC)
        self.bfc0 = nn.Linear(64,hidden_dims[0])
        self.bfc1 = nn.Linear(64,hidden_dims[0])
        self.bfc2 = nn.Linear(hidden_dims[0],hidden_dims[1])
        self.betsize_value_output = nn.Linear(64,1)
        self.betsize_advantage_output = nn.Linear(64,self.nA)
        
    def forward(self,obs):
        x = obs
        M,c = x.size()
        hand = x[0,self.mapping['observation']['rank']].long().unsqueeze(0)
        vil_hand = x[0,self.mapping['observation']['vil_rank']].long().unsqueeze(0)
        hands = torch.cat([hand,vil_hand],dim=-1)

        hot_ranks = self.one_hot_kuhn[hands.long()]
        if hot_ranks.dim() == 2:
            hot_ranks = hot_ranks.unsqueeze(0)
        last_action = x[:,self.mapping['observation']['previous_action']].long()
        last_betsize = x[:,self.mapping['observation']['previous_betsize']].float().unsqueeze(1)
        a1 = self.action_emb(last_action)

        h = self.conv(hot_ranks.float())
        h = h.view(-1).unsqueeze(0).repeat(M,1)
        x = torch.cat([h,a1,last_betsize],dim=-1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        q_input = x.view(M,-1)
        a = self.advantage_output(q_input)
        v = self.value_output(q_input)
        v = v.expand_as(a)
        q = v + a - a.mean(1,keepdim=True).expand_as(a)

        # Could only do a forward pass if betsizes are available
        x = self.activation(self.bfc0(x))
        x = self.activation(self.bfc1(x))
        x = self.activation(self.bfc2(x))
        betsize_input = x.view(M,-1)
        ab = self.betsize_advantage_output(betsize_input)
        vb = self.betsize_value_output(betsize_input)
        vb = vb.expand_as(ab)
        qb = vb + ab - ab.mean(1,keepdim=True).expand_as(ab)

        outputs = {'value':q,'betsize':qb}
        return outputs

################################################
#            Flat Betsize Networks             #
################################################

class FlatBetsizeActor(nn.Module):
    def __init__(self,seed,nS,nA,nB,params,hidden_dims=(64,64),activation=F.leaky_relu):
        """
        Num Categories: nA (check,fold,call,bet,raise)
        Num Betsizes: nB (various betsizes)
        """
        super().__init__()
        self.activation = activation
        self.nS = nS
        self.nA = nA
        self.nB = nB
        self.combined_output = nA - 2 + nB
        self.helper_functions = NetworkFunctions(self.nA,self.nB)
        
        self.seed = torch.manual_seed(seed)
        self.mapping = params['mapping']
        self.hand_emb = Embedder(5,64)
        self.action_emb = Embedder(6,64)
        self.betsize_emb = Embedder(self.nB,64)
        self.noise = GaussianNoise()
        self.fc1 = nn.Linear(129,hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0],hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1],self.combined_output)
        
    def forward(self,state,action_mask,betsize_mask):
        mask = combined_masks(action_mask,betsize_mask)
        x = state
        hand = x[:,self.mapping['state']['rank']].long()
        last_action = x[:,self.mapping['state']['previous_action']].long()
        previous_betsize = x[:,self.mapping['state']['previous_betsize']].float()
        if previous_betsize.dim() == 1:
            previous_betsize = previous_betsize.unsqueeze(1)
        hand = self.hand_emb(hand)
        last_action_emb = self.action_emb(last_action)
        # print('hand,last_action_emb,previous_betsize',hand.size(),last_action_emb.size(),previous_betsize.size())
        x = torch.cat([hand,last_action_emb,previous_betsize],dim=-1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        cateogry_logits = self.fc3(x)
        cateogry_logits = self.noise(cateogry_logits)
        action_soft = F.softmax(cateogry_logits,dim=-1)
        # print(action_soft.size(),mask.size())
        action_probs = norm_frequencies(action_soft,mask)
        # action_probs = action_probs * mask
        # action_probs /= torch.sum(action_probs)
        m = Categorical(action_probs)
        action = m.sample()

        action_category,betsize_category = self.helper_functions.unwrap_action(action,last_action)
        # print('state',state)
        # print('action_category,betsize_category',action_category,betsize_category)
        
        outputs = {
            'action':action,
            'action_category':action_category,
            'action_prob':m.log_prob(action),
            'action_probs':action_probs,
            'betsize':betsize_category
            }
        return outputs

class FlatBetsizeCritic(nn.Module):
    def __init__(self,seed,nS,nA,nB,params,hidden_dims=(64,64),activation=F.leaky_relu):
        super().__init__()
        self.activation = activation
        self.nS = nS
        self.nA = nA
        self.nB = nB
        self.combined_output = nA - 2 + nB
        
        self.seed = torch.manual_seed(seed)
        self.use_embedding = params['embedding']
        self.mapping = params['mapping']
        self.one_hot_kuhn = torch.nn.functional.one_hot(torch.arange(0,4))
        self.one_hot_actions = torch.nn.functional.one_hot(torch.arange(0,6))
        self.hand_emb = Embedder(5,32)
        self.action_emb = Embedder(6,32)
        self.positional_embeddings = Embedder(2,32)

        self.conv = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=3, stride=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )
        self.fc0 = nn.Linear(64,hidden_dims[0])
        self.fc1 = nn.Linear(97,hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0],hidden_dims[1])
        self.value_output = nn.Linear(64,1)
        self.advantage_output = nn.Linear(64,self.combined_output)
        
    def forward(self,obs):
        x = obs
        M,c = x.size()
        hand = x[0,self.mapping['observation']['rank']].long().unsqueeze(0)
        vil_hand = x[0,self.mapping['observation']['vil_rank']].long().unsqueeze(0)
        hands = torch.cat([hand,vil_hand],dim=-1)
        hot_ranks = self.one_hot_kuhn[hands.long()]
        if hot_ranks.dim() == 2:
            hot_ranks = hot_ranks.unsqueeze(0)
        last_action = x[:,self.mapping['observation']['previous_action']].long()
        last_betsize = x[:,self.mapping['observation']['previous_betsize']].float()
        if last_betsize.dim() == 1:
            last_betsize = last_betsize.unsqueeze(1)
        a1 = self.action_emb(last_action)

        # print(hot_ranks.size())
        h = self.conv(hot_ranks.float())
        h = h.view(-1).unsqueeze(0).repeat(M,1)
        # print('h,a1,last_betsize',h.size(),a1.size(),last_betsize.size())
        x = torch.cat([h,a1,last_betsize],dim=-1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        q_input = x.view(M,-1)
        a = self.advantage_output(q_input)
        v = self.value_output(q_input)
        v = v.expand_as(a)
        q = v + a - a.mean(1,keepdim=True).expand_as(a)

        outputs = {'value':q }
        return outputs

################################################
#            Normal Kuhn Networks              #
################################################

class Baseline(nn.Module):
    def __init__(self,seed,nS,nC,nA,params,hidden_dims=(64,64),activation=F.leaky_relu):
        super().__init__()
        self.activation = activation
        self.nS = nS
        self.nC = nC
        self.nA = nA
        
        self.seed = torch.manual_seed(seed)
        self.mapping = params['mapping']
        self.hand_emb = Embedder(5,64)
        self.action_emb = Embedder(6,64)
        self.noise = GaussianNoise()
        self.fc1 = nn.Linear(64+64,hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0],hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1],nC)
        
    def forward(self,state,mask):
        x = state
        if not isinstance(state,torch.Tensor):
            x = torch.tensor(x,dtype=torch.float32) #device = self.device,
            x = x.unsqueeze(0)
        # print(x)
        # print(self.mapping['state']['rank'])
        # print(self.mapping['state']['previous_action'])
        # print(x[:,self.mapping['state']['rank']])
        # print(x[:,self.mapping['state']['previous_action']])
        hand = x[:,self.mapping['state']['rank']].long()
        last_action = x[:,self.mapping['state']['previous_action']].long()
        hand = self.hand_emb(hand)
        last_action = self.action_emb(last_action)
        x = torch.cat([hand,last_action],dim=-1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        action_logits = self.noise(x)
        action_soft = F.softmax(action_logits,dim=-1)
        action_probs = norm_frequencies(action_soft,mask)
        m = Categorical(action_probs)
        action = m.sample()

        outputs = {
            'action':action,
            'action_prob':m.log_prob(action),
            'action_probs':action_probs}
        return outputs


class BaselineKuhnCritic(nn.Module):
    def __init__(self,seed,nS,nC,nA,params,hidden_dims=(64,64),activation=F.leaky_relu):
        super().__init__()
        self.activation = activation
        self.nS = nS
        self.nC = nC
        self.nA = nA
        
        self.seed = torch.manual_seed(seed)
        self.use_embedding = params['embedding']
        self.mapping = params['mapping']
        self.one_hot_kuhn = torch.nn.functional.one_hot(torch.arange(0,4))
        self.one_hot_actions = torch.nn.functional.one_hot(torch.arange(0,6))
        self.hand_emb = Embedder(5,32)
        self.action_emb = Embedder(6,32)
        self.positional_embeddings = Embedder(2,32)

        self.conv = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=3, stride=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.action_conv = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=3, stride=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        # self.lstm = nn.LSTM(96, 32)
        self.fc0 = nn.Linear(64,hidden_dims[0])
        self.fc1 = nn.Linear(128,hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0],hidden_dims[1])
        self.value_output = nn.Linear(64,1)
        # self.q_values = nn.Linear(hidden_dims[1],self.nA)
        
    def forward(self,obs,action):
        x = obs
        M,c = x.size()
        hand = x[:,self.mapping['observation']['rank']].long()
        vil_hand = x[:,self.mapping['observation']['vil_rank']].long()
        hands = torch.cat([hand,vil_hand],dim=-1)
        last_action = x[:,self.mapping['observation']['previous_action']].long()
        hot_ranks = self.one_hot_kuhn[hands.long()]
        if hot_ranks.dim() == 2:
            hot_ranks = hot_ranks.unsqueeze(0)

        # Convolve actions
        # hot_prev_action = self.one_hot_actions[last_action]
        # hot_cur_action = self.one_hot_actions[action]
        # actions = torch.stack((hot_prev_action,hot_cur_action)).permute(1,0,2)

        # Embed actions
        positions = torch.arange(2)
        a1 = self.action_emb(last_action)
        a2 = self.action_emb(action)
        p1 = self.positional_embeddings(positions[0])
        p2 = self.positional_embeddings(positions[1])

        a1 += p1
        a2 += p2

        h = self.conv(hot_ranks.float()).view(M,-1)
        x = torch.cat([h,a2,a1],dim=-1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = x.view(M,-1)

        outputs = {
            'value':self.value_output(x)
            }
        return outputs
        
class BaselineCritic(nn.Module):
    def __init__(self,seed,nS,nC,nA,params,hidden_dims=(64,64),activation=F.leaky_relu):
        super().__init__()
        self.activation = activation
        self.nS = nS
        self.nC = nC
        self.nA = nA
        
        self.seed = torch.manual_seed(seed)
        self.use_embedding = params['embedding']
        self.mapping = params['mapping']
        self.one_hot_kuhn = torch.nn.functional.one_hot(torch.arange(0,4))
        self.one_hot_actions = torch.nn.functional.one_hot(torch.arange(0,6))
        self.hand_emb = Embedder(5,32)
        self.action_emb = Embedder(6,32)
        self.positional_embeddings = Embedder(2,32)

        self.conv = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=3, stride=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )
        self.fc0 = nn.Linear(64,hidden_dims[0])
        self.fc1 = nn.Linear(96,hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0],hidden_dims[1])
        self.value_output = nn.Linear(64,1)
        self.advantage_output = nn.Linear(64,self.nC)
        
    def forward(self,obs):
        x = obs
        M,c = x.size()
        hand = x[:,self.mapping['observation']['rank']].long()
        vil_hand = x[:,self.mapping['observation']['vil_rank']].long()
        hands = torch.stack((hand,vil_hand)).permute(1,0)
        last_action = x[:,self.mapping['observation']['previous_action']].long()
        a1 = self.action_emb(last_action)
        hot_ranks = self.one_hot_kuhn[hands.long()]
        if hot_ranks.dim() == 2:
            hot_ranks = hot_ranks.unsqueeze(0)
        h = self.conv(hot_ranks.float()).view(M,-1)
        x = torch.cat([h,a1],dim=-1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = x.view(M,-1)
        a = self.advantage_output(x)
        v = self.value_output(x)
        v = v.expand_as(a)
        q = v + a - a.mean(1,keepdim=True).expand_as(a)
        outputs = {
            'value':q
            }
        return outputs

"""
Dueling QNetwork for function aproximation. Splits the network prior to the end into two streams V and Q. 
V is the estimate of the value of the state. Q is the advantage of each action given the state.
Two formulations for subtracting Q from V:
V - max(Q)
This verision makes more sense theoretically as the value of V* should equal the max(Q*(s,A)). 
But in practice mean allows for better performance.
V - mean(Q)
Same as max except now they are separated by a constant. 
And not as susceptable to over optimism due to randomness of Q values.
"""

    
class Dueling_QNetwork(nn.Module):
    def __init__(self,seed,state_space,action_space,hidden_dims=(32,32),activation_fc=F.relu):
        super(Dueling_QNetwork,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.activation_fc = activation_fc
        self.seed = torch.manual_seed(seed)
        print('hidden_dims',hidden_dims)
        self.input_layer = nn.Linear(state_space,hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i],hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        self.value_output = nn.Linear(hidden_dims[-1],1)
        self.advantage_output = nn.Linear(hidden_dims[-1],action_space)
        
    def forward(self,state):
        x = state
        if not isinstance(state,torch.Tensor):
            x = torch.tensor(x,dtype=torch.float32,device = self.device,)
            x = x.unsqueeze(0)
        x = self.activation_fc(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        a = self.advantage_output(x)
        v = self.value_output(x)
        v = v.expand_as(a)
        q = v + a - a.mean(1,keepdim=True).expand_as(a)
        return q

################################################
#           13 card winner prediction          #
################################################

class ThirteenCard(nn.Module):
    def __init__(self,params,hidden_dims=(15,32),activation_fc=F.relu):
        super().__init__()
        self.params = params
        self.nA = params['nA']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.activation_fc = activation_fc
        self.seed = torch.manual_seed(params['seed'])
        # Input is (1,13,2) -> (1,13,64)
        self.one_hot_suits = torch.nn.functional.one_hot(torch.arange(0,dt.SUITS.HIGH))
        self.one_hot_ranks = torch.nn.functional.one_hot(torch.arange(0,dt.RANKS.HIGH))
        # Input is (b,4,2) -> (b,4,4) and (b,4,13)
        self.suit_conv = nn.Sequential(
            nn.Conv1d(13, 128, kernel_size=1, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.rank_conv = nn.Sequential(
            nn.Conv1d(13, 128, kernel_size=5, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.hidden_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i],hidden_dims[i+1]))
            self.bn_layers.append(nn.BatchNorm1d(128))
        self.dropout = nn.Dropout(0.5)
        self.categorical_output = nn.Linear(4096,self.nA)

    def forward(self,x):
        # Input is (b,13,2)
        M,c,h = x.size()
        ranks = x[:,:,0].long()
        suits = x[:,:,1].long()

        hot_ranks = self.one_hot_ranks[ranks]
        hot_suits = self.one_hot_suits[suits]

        s = self.suit_conv(hot_suits.float())
        r = self.rank_conv(hot_ranks.float())
        x = torch.cat((r,s),dim=-1)
        # should be (b,64,88)

        for i,hidden_layer in enumerate(self.hidden_layers):
            x = self.activation_fc(self.bn_layers[i](hidden_layer(x)))
        x = x.view(M,-1)
        x = self.dropout(x)
        return torch.tanh(self.categorical_output(x))
        
# Emb + fc
class ThirteenCardV2(nn.Module):
    def __init__(self,params,hidden_dims=(15,64,32),activation_fc=F.relu):
        super().__init__()
        self.params = params
        self.nA = params['nA']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.activation_fc = activation_fc
        self.seed = torch.manual_seed(params['seed'])
        # Input is (1,13,2) -> (1,13,64)
        self.one_hot_suits = torch.nn.functional.one_hot(torch.arange(0,dt.SUITS.HIGH))
        self.one_hot_ranks = torch.nn.functional.one_hot(torch.arange(0,dt.RANKS.HIGH))
        # Input is (b,4,2) -> (b,4,4) and (b,4,13)
        self.suit_conv = nn.Sequential(
            nn.Conv1d(9, 64, kernel_size=1, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.rank_conv = nn.Sequential(
            nn.Conv1d(9, 64, kernel_size=5, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.hidden_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i],hidden_dims[i+1]))
            self.bn_layers.append(nn.BatchNorm1d(64))
        self.dropout = nn.Dropout(0.5)
        self.categorical_output = nn.Linear(2048,self.nA)

    def forward(self,x):
        # Input is (b,13,2)
        M,c,h = x.size()
        ranks = x[:,:,0].long()
        suits = x[:,:,1].long()

        hot_ranks = self.one_hot_ranks[ranks]
        # (b,13,13)
        hot_suits = self.one_hot_suits[suits]
        # (b,13,4)
        hero_board_ranks = hot_ranks[:,torch.tensor([0,1,2,3,8,9,10,11,12])]
        hero_board_suits = hot_suits[:,torch.tensor([0,1,2,3,8,9,10,11,12])]
        vil_board_ranks = hot_ranks[:,4:]
        vil_board_suits = hot_suits[:,4:]
        s = self.rank_conv(hero_board_ranks.float())
        r = self.suit_conv(hero_board_suits.float())
        x1 = torch.cat((r,s),dim=-1)
        # should be (b,64,88)
        s2 = self.rank_conv(vil_board_ranks.float())
        r2 = self.suit_conv(vil_board_suits.float())
        x2 = torch.cat((r2,s2),dim=-1)

        x = x1 - x2
        for i,hidden_layer in enumerate(self.hidden_layers):
            x = self.activation_fc(self.bn_layers[i](hidden_layer(x)))
        x = x.view(M,-1)
        x = self.dropout(x)
        return torch.tanh(self.categorical_output(x))

# Conv + multiheaded attention
class ThirteenCardV3(nn.Module):
    def __init__(self,params,hidden_dims=(64,32),activation_fc=F.relu):
        super().__init__()
        self.params = params
        self.nA = params['nA']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.activation_fc = activation_fc
        self.seed = torch.manual_seed(params['seed'])
        # Input is (1,13,2) -> (1,13,64)
        self.hidden_conv_layers = nn.ModuleList()
        self.hidden_bn_layers = nn.ModuleList()
        for i in range(params['conv_layers']):
            self.conv = nn.Conv1d(2, 64, kernel_size=13, stride=1)
            self.hidden_conv_layers.append(self.conv)
            if params['batchnorm'] == True:
                self.bn = nn.BatchNorm1d(64)
                self.hidden_bn_layers.append(self.bn)
            # Output shape is (1,64,9,4,4)
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i],hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        self.value_output = nn.Linear(hidden_dims[-1],self.nA)

    def forward(self,state):
        # Split input into the 60 combinations for both hero+villain
        x = state
        if not isinstance(state,torch.Tensor):
            x = torch.tensor(x,dtype=torch.float32,device = self.device)
            # x = x.unsqueeze(0)
        M = x.size(0)
        for i,layer in enumerate(self.hidden_conv_layers):
            if len(self.hidden_bn_layers):
                x = self.activation_fc(self.hidden_bn_layers[i](layer(x)))
            else:
                x = self.activation_fc(layer(x))
        x = x.view(M,-1)
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        v = torch.tanh(self.value_output(x))
        return v

################################################
#           Hand type categorization           #
################################################

################################################
#           Nine Card categorization           #
################################################

class HandClassification(nn.Module):
    def __init__(self,params,hidden_dims=(15,32,32),activation_fc=F.relu):
        super(HandClassification,self).__init__()
        self.params = params
        self.nA = params['nA']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.activation_fc = activation_fc
        self.seed = torch.manual_seed(params['seed'])
        
        self.one_hot_suits = torch.nn.functional.one_hot(torch.arange(0,dt.SUITS.HIGH))
        self.one_hot_ranks = torch.nn.functional.one_hot(torch.arange(0,dt.RANKS.HIGH))

        # Input is (b,4,2) -> (b,4,4) and (b,4,13)
        self.suit_conv = nn.Sequential(
            nn.Conv1d(9, 64, kernel_size=1, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.rank_conv = nn.Sequential(
            nn.Conv1d(9, 64, kernel_size=5, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.hidden_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i],hidden_dims[i+1]))
            self.bn_layers.append(nn.BatchNorm1d(64))
        self.dropout = nn.Dropout(0.5)
        self.categorical_output = nn.Linear(2048,self.nA)

    def forward(self,x):
        # Input is (b,9,2)
        M,c,h = x.size()
        ranks = x[:,:,0].long()
        suits = x[:,:,1].long()

        hot_ranks = self.one_hot_ranks[ranks]
        hot_suits = self.one_hot_suits[suits]

        s = self.suit_conv(hot_suits.float())
        r = self.rank_conv(hot_ranks.float())
        x = torch.cat((r,s),dim=-1)
        # should be (b,64,88)

        for i,hidden_layer in enumerate(self.hidden_layers):
            x = self.activation_fc(self.bn_layers[i](hidden_layer(x)))
        x = x.view(M,-1)
        x = self.dropout(x)
        return self.categorical_output(x)

# Emb + fc
class HandClassificationV2(nn.Module):
    def __init__(self,params,hidden_dims=(64,64,32),activation_fc=F.leaky_relu):
        super(HandClassificationV2,self).__init__()
        self.params = params
        self.nA = params['nA']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.activation_fc = activation_fc
        self.seed = torch.manual_seed(params['seed'])
        
        self.rank_emb = Embedder(15,32)
        self.suit_emb = Embedder(4,32)
        self.pos_emb = nn.Embedding(9, 32)
        self.hidden_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i],hidden_dims[i+1]))
            self.bn_layers.append(nn.BatchNorm1d(9))
        self.dropout = nn.Dropout(0.5)
        self.categorical_output = nn.Linear(288,self.nA)

    def forward(self,x):
        if not isinstance(x,torch.Tensor):
            x = torch.tensor(x,dtype=torch.float32,device = self.device)
            # x = x.unsqueeze(0)
        # generate position embeddings
        ranks = self.rank_emb(x[:,:,0].long())
        suits = self.suit_emb(x[:,:,1].long())

        b, t, k = ranks.size()
        positions = torch.arange(t)
        positions = self.pos_emb(positions)[None, :, :].expand(b, t, k)
        ranks += positions
        suits += positions
        x = torch.cat((ranks,suits),dim=-1)
        # Flatten layer but retain number of samples
        for i,hidden_layer in enumerate(self.hidden_layers):
            x = self.activation_fc(self.bn_layers[i](hidden_layer(x)))
        x = x.view(b,-1)
        x = self.dropout(x)
        action_logits = self.categorical_output(x)
        return action_logits

"""
Comparing unspooling the hand into a (60,5) vector 
"""
class HandClassificationV3(nn.Module):
    def __init__(self,params,hidden_dims=(44,32,32),activation_fc=F.relu):
        super(HandClassificationV3,self).__init__()
        self.params = params
        self.nA = params['nA']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.activation_fc = activation_fc
        self.seed = torch.manual_seed(params['seed'])
        # Input is (1,13,2) -> (1,13,64)
        self.rank_emb = Embedder(15,32)
        self.suit_emb = Embedder(4,12)
        # Output shape is (1,64,9,4,4)
        self.hidden_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i],hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
            self.bn_layers.append(nn.BatchNorm2d(60))
        self.dropout = nn.Dropout(0.5)
        self.categorical_output = nn.Linear(9600,self.nA)

    def forward(self,state):
        # Input is M,60,5,2
        x = state
        if not isinstance(state,torch.Tensor):
            x = torch.tensor(x,dtype=torch.float32,device = self.device)
            # x = x.unsqueeze(0)
        M = x.size(0)
        ranks = self.rank_emb(x[:,:,:,0].long())
        suits = self.suit_emb(x[:,:,:,1].long())
        x = torch.cat((ranks,suits),dim=-1)
        for i,hidden_layer in enumerate(self.hidden_layers):
            x = self.activation_fc(hidden_layer(x))
            # x = self.activation_fc(self.bn_layers[i](hidden_layer(x)))
        # Flatten layer but retain number of samples
        x = x.view(M,-1) # * x.shape[2] * x.shape[3] * x.shape[4])
        # x = self.dropout(x)
        return self.categorical_output(x)


"""
Takes a (60,5) vector, Uses an attention mechanism to select the hand it thinks is most likely.
"""
class HandClassificationV4(nn.Module):
    def __init__(self,params,hidden_dims=(44,32,32),activation_fc=F.relu):
        super(HandClassificationV4,self).__init__()
        self.params = params
        self.nA = params['nA']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.activation_fc = activation_fc
        self.seed = torch.manual_seed(params['seed'])
        # Input is (1,13,2) -> (1,13,64)
        self.rank_emb = Embedder(15,32)
        self.suit_emb = Embedder(4,12)
        # Output shape is (1,64,9,4,4)
        self.hidden_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i],hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
            self.bn_layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
        self.attention = WideAttention(2,160)
        self.categorical_output = nn.Linear(9600,self.nA)

    def forward(self,state):
        x = state
        if not isinstance(state,torch.Tensor):
            x = torch.tensor(x,dtype=torch.float32,device = self.device)
            # x = x.unsqueeze(0)
        M = x.size(0)
        ranks = self.rank_emb(x[:,:,:,0].long())
        suits = self.suit_emb(x[:,:,:,1].long())
        x = torch.cat((ranks,suits),dim=-1)
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        # Flatten layer but retain number of samples
        b,m,t,k = x.size()
        x = x.view(b,m,t*k)
        x = self.attention(x)
        x = x.view(M,-1)    
        return self.categorical_output(x)


################################################
#           fivecard categorization            #
################################################

class FiveCardClassification(nn.Module):
    def __init__(self,params,hidden_dims=(32,32,16),activation_fc=F.relu):
        super().__init__()
        self.params = params
        self.nA = params['nA']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.activation_fc = activation_fc
        self.seed = torch.manual_seed(params['seed'])

        self.one_hot_suits = torch.nn.functional.one_hot(torch.arange(0,dt.SUITS.HIGH))
        self.one_hot_ranks = torch.nn.functional.one_hot(torch.arange(0,dt.RANKS.HIGH))

        # Input is (b,5,2) -> (b,5,4)
        self.suit_conv = nn.Conv1d(5, 64, kernel_size=1, stride=1)
        self.suit_bn = nn.BatchNorm1d(64)
        # Output shape is (b,64,1) 
        # Input is (b,5,2) -> (b,5,13) or (b,5,15)
        self.rank_conv = nn.Conv1d(5, 64, kernel_size=5, stride=1)
        self.rank_bn = nn.BatchNorm1d(64)
        # Output shape is (b,64,9) or (b,64,11) 
        self.rank_output = nn.Linear(11,32)
        self.suit_output = nn.Linear(4,12)
        self.hidden_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i],hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
            self.bn_layers.append(nn.BatchNorm1d(64))
        self.dropout = nn.Dropout(0.5)
        self.categorical_output = nn.Linear(1792,self.nA)

    def forward(self,x):
        # Input is M,5,2
        assert(isinstance(x,torch.Tensor))
        M = x.size(0)
        ranks = x[:,:,0]
        suits = x[:,:,1]
        hot_ranks = self.one_hot_ranks[ranks.long()]
        hot_suits = self.one_hot_suits[suits.long()]

        s = self.activation_fc(self.suit_bn(self.suit_conv(hot_suits.float())))
        s = self.activation_fc(self.suit_output(s))

        r = self.activation_fc(self.rank_bn(self.rank_conv(hot_ranks.float())))
        r = self.activation_fc(self.rank_output(r))
        for i,hidden_layer in enumerate(self.hidden_layers):
            # r = self.activation_fc(hidden_layer(r))
            r = self.activation_fc(self.bn_layers[i](hidden_layer(r)))
        r = r.view(M,-1)
        s = s.view(M,-1)
        x = torch.cat((s,r),dim=-1)
        # x = self.dropout(x) 
        return self.categorical_output(x)

# 3d conv2d
class FiveCardClassificationV2(nn.Module):
    def __init__(self,params,hidden_dims=(32,32,16),activation_fc=F.relu):
        super().__init__()
        self.params = params
        self.nA = params['nA']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.activation_fc = activation_fc
        self.seed = torch.manual_seed(params['seed'])
        # Input is (b,5,2) -> (b,5,13) or (b,5,15)
        self.rank_conv = nn.Conv2d(5, 64, kernel_size=5, stride=1)
        self.rank_bn = nn.BatchNorm2d(64)
        # Output shape is (b,64,9) or (b,64,11) 
        self.rank_output = nn.Linear(10,32)
        self.suit_output = nn.Linear(4,12)
        self.hidden_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i],hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
            self.bn_layers.append(nn.BatchNorm1d(64))
        self.dropout = nn.Dropout(0.5)
        self.categorical_output = nn.Linear(1792,self.nA)

    def forward(self,x):
        # Input is M,5,2
        assert(isinstance(x,torch.Tensor))
        M = x.size(0)
        ranks = x[:,:,0]
        suits = x[:,:,1]
        three_d = torch.zeros(M,5,dt.RANKS.HIGH,dt.SUITS.HIGH)
        # (b,5,13)
        for j in range(M):
            for i in range(5):
                three_d[j,i,ranks[j,i],suits[j,i]] = 1
        r = self.activation_fc(self.rank_bn(self.rank_conv(three_d.float())))
        r = self.activation_fc(self.rank_output(r))
        for i,hidden_layer in enumerate(self.hidden_layers):
            # r = self.activation_fc(hidden_layer(r))
            r = self.activation_fc(self.bn_layers[i](hidden_layer(r)))
        r = r.view(M,-1)
        return self.categorical_output(x)

################################################
#           tencard categorization             #
################################################

# Convolve each hand separately
class TenCardClassification(nn.Module):
    def __init__(self,params,hidden_dims=(32,32,16),activation_fc=F.relu):
        super().__init__()
        self.params = params
        self.nA = params['nA']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.activation_fc = activation_fc
        self.seed = torch.manual_seed(params['seed'])

        self.one_hot_suits = torch.nn.functional.one_hot(torch.arange(0,dt.SUITS.HIGH))
        self.one_hot_ranks = torch.nn.functional.one_hot(torch.arange(0,dt.RANKS.HIGH))

        # Input is (b,5,2) -> (b,5,4)
        self.suit_conv = nn.Conv1d(5, 64, kernel_size=1, stride=1)
        self.suit_bn = nn.BatchNorm1d(64)
        # Output shape is (b,64,1) 
        # Input is (b,5,2) -> (b,5,13) or (b,5,15)
        self.rank_conv = nn.Conv1d(5, 64, kernel_size=5, stride=1)
        self.rank_bn = nn.BatchNorm1d(64)
        # Output shape is (b,64,9) or (b,64,11) 
        self.rank_output = nn.Linear(10,32)
        self.suit_output = nn.Linear(4,12)
        self.hidden_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i],hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
            self.bn_layers.append(nn.BatchNorm1d(64))
        self.dropout = nn.Dropout(0.5)
        self.output = nn.Linear(1792,self.nA)

    def forward(self,x):
        # Input is M,10,2
        assert(isinstance(x,torch.Tensor))
        M = x.size(0)
        ranks1 = x[:,:5,0]
        suits1 = x[:,:5,1]
        hot_ranks1 = self.one_hot_ranks[ranks1.long()]
        hot_suits1 = self.one_hot_suits[suits1.long()]
        ranks2 = x[:,5:,0]
        suits2 = x[:,5:,1]
        hot_ranks2 = self.one_hot_ranks[ranks2.long()]
        hot_suits2 = self.one_hot_suits[suits2.long()]

        s1 = self.activation_fc(self.suit_bn(self.suit_conv(hot_suits1.float())))
        s1 = self.activation_fc(self.suit_output(s1))

        s2 = self.activation_fc(self.suit_bn(self.suit_conv(hot_suits2.float())))
        s2 = self.activation_fc(self.suit_output(s2))

        r1 = self.activation_fc(self.rank_bn(self.rank_conv(hot_ranks1.float())))
        r1 = self.activation_fc(self.rank_output(r1))

        r2 = self.activation_fc(self.rank_bn(self.rank_conv(hot_ranks2.float())))
        r2 = self.activation_fc(self.rank_output(r2))

        for i,hidden_layer in enumerate(self.hidden_layers):
            # r = self.activation_fc(hidden_layer(r))
            r1 = self.activation_fc(self.bn_layers[i](hidden_layer(r1)))

        for i,hidden_layer in enumerate(self.hidden_layers):
            # r = self.activation_fc(hidden_layer(r))
            r2 = self.activation_fc(self.bn_layers[i](hidden_layer(r2)))

        r1 = r1.view(M,-1)
        s1 = s1.view(M,-1)
        x1 = torch.cat((s1,r1),dim=-1)

        r2 = r2.view(M,-1)
        s2 = s2.view(M,-1)
        x2 = torch.cat((s2,r2),dim=-1)

        # x = self.dropout(x) 
        return torch.tanh(self.output(x1 - x2))

# Convolving everything at once
class TenCardClassificationV2(nn.Module):
    def __init__(self,params,hidden_dims=(32,32,16),activation_fc=F.relu):
        super().__init__()
        self.params = params
        self.nA = params['nA']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.activation_fc = activation_fc
        self.seed = torch.manual_seed(params['seed'])

        self.one_hot_suits = torch.nn.functional.one_hot(torch.arange(0,dt.SUITS.HIGH))
        self.one_hot_ranks = torch.nn.functional.one_hot(torch.arange(0,dt.RANKS.HIGH))

        # Input is (b,5,2) -> (b,5,4)
        self.suit_conv = nn.Conv1d(10, 64, kernel_size=1, stride=1)
        self.suit_bn = nn.BatchNorm1d(64)
        # Output shape is (b,64,1) 
        # Input is (b,5,2) -> (b,5,13) or (b,5,15)
        self.rank_conv = nn.Conv1d(10, 64, kernel_size=5, stride=1)
        self.rank_bn = nn.BatchNorm1d(64)
        # Output shape is (b,64,9) or (b,64,11) 
        self.rank_output = nn.Linear(11,32)
        self.suit_output = nn.Linear(4,12)
        self.hidden_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i],hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
            self.bn_layers.append(nn.BatchNorm1d(64))
        self.dropout = nn.Dropout(0.5)
        self.categorical_output = nn.Linear(1792,self.nA)

    def forward(self,x):
        # Input is M,5,2
        assert(isinstance(x,torch.Tensor))
        M = x.size(0)
        ranks = x[:,:,0]
        suits = x[:,:,1]
        hot_ranks = self.one_hot_ranks[ranks.long()]
        hot_suits = self.one_hot_suits[suits.long()]

        s = self.activation_fc(self.suit_bn(self.suit_conv(hot_suits.float())))
        s = self.activation_fc(self.suit_output(s))

        r = self.activation_fc(self.rank_bn(self.rank_conv(hot_ranks.float())))
        r = self.activation_fc(self.rank_output(r))
        for i,hidden_layer in enumerate(self.hidden_layers):
            # r = self.activation_fc(hidden_layer(r))
            r = self.activation_fc(self.bn_layers[i](hidden_layer(r)))
        r = r.view(M,-1)
        s = s.view(M,-1)
        x = torch.cat((s,r),dim=-1)
        # x = self.dropout(x) 
        return torch.tanh(self.categorical_output(x))

# Using 3d convolutions (2d with channels)
class TenCardClassificationV3(nn.Module):
    def __init__(self,params,hidden_dims=(32,32,16),activation_fc=F.relu):
        super().__init__()
        self.params = params
        self.nA = params['nA']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.activation_fc = activation_fc
        self.seed = torch.manual_seed(params['seed'])

        self.one_hot_suits = torch.nn.functional.one_hot(torch.arange(0,dt.SUITS.HIGH))
        self.one_hot_ranks = torch.nn.functional.one_hot(torch.arange(0,dt.RANKS.HIGH))

        # Input is (b,5,2) -> (b,5,13) or (b,5,15)
        self.rank_conv = nn.Conv2d(5, 64, kernel_size=(5,4), stride=(1,1))
        self.rank_bn = nn.BatchNorm2d(64)
        # Output shape is (b,64,9) or (b,64,11) 
        self.rank_output = nn.Linear(10,32)
        self.suit_output = nn.Linear(4,12)
        self.hidden_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i],hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
            self.bn_layers.append(nn.BatchNorm1d(64))
        self.dropout = nn.Dropout(0.5)
        self.categorical_output = nn.Linear(1792,self.nA)

    def forward(self,x):
        # Input is M,5,2
        assert(isinstance(x,torch.Tensor))
        M = x.size(0)
        ranks = x[:,:,0]
        suits = x[:,:,1]
        hot_ranks = self.one_hot_ranks[ranks.long()]
        hot_suits = self.one_hot_suits[suits.long()]

        s = self.activation_fc(self.suit_bn(self.suit_conv(hot_suits.float())))
        s = self.activation_fc(self.suit_output(s))

        r = self.activation_fc(self.rank_bn(self.rank_conv(hot_ranks.float())))
        r = self.activation_fc(self.rank_output(r))
        for i,hidden_layer in enumerate(self.hidden_layers):
            # r = self.activation_fc(hidden_layer(r))
            r = self.activation_fc(self.bn_layers[i](hidden_layer(r)))
        r = r.view(M,-1)
        s = s.view(M,-1)
        x = torch.cat((s,r),dim=-1)
        # x = self.dropout(x) 
        return self.categorical_output(x)

################################################
#           Blockers categorization            #
################################################

class BlockerClassification(nn.Module):
    def __init__(self,params,hidden_dims=(15,32,32),activation_fc=F.relu):
        super().__init__()
        self.params = params
        self.nA = params['nA']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.activation_fc = activation_fc
        self.seed = torch.manual_seed(params['seed'])
        
        self.one_hot_suits = torch.nn.functional.one_hot(torch.arange(0,dt.SUITS.HIGH))
        self.one_hot_ranks = torch.nn.functional.one_hot(torch.arange(0,dt.RANKS.HIGH))

        # Input is (b,4,2) -> (b,4,4) and (b,4,13)
        self.suit_conv = nn.Sequential(
            nn.Conv1d(9, 64, kernel_size=1, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.rank_conv = nn.Sequential(
            nn.Conv1d(9, 64, kernel_size=5, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.hidden_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i],hidden_dims[i+1]))
            self.bn_layers.append(nn.BatchNorm1d(64))
        self.dropout = nn.Dropout(0.5)
        self.categorical_output = nn.Linear(2048,self.nA)

    def forward(self,x):
        # Input is (b,9,2)
        M,c,h = x.size()
        ranks = x[:,:,0].long()
        suits = x[:,:,1].long()

        hot_ranks = self.one_hot_ranks[ranks]
        hot_suits = self.one_hot_suits[suits]

        s = self.suit_conv(hot_suits.float())
        r = self.rank_conv(hot_ranks.float())
        x = torch.cat((r,s),dim=-1)
        # should be (b,64,88)

        for i,hidden_layer in enumerate(self.hidden_layers):
            x = self.activation_fc(self.bn_layers[i](hidden_layer(x)))
        x = x.view(M,-1)
        x = self.dropout(x)
        return torch.sigmoid(self.categorical_output(x))

################################################
#           Hand Rank categorization           #
################################################

class HandRankClassification(nn.Module):
    def __init__(self,params,hidden_dims=(15,32,32),activation_fc=F.relu):
        super().__init__()
        self.params = params
        self.nA = params['nA']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.activation_fc = activation_fc
        self.seed = torch.manual_seed(params['seed'])
        
        self.one_hot_suits = torch.nn.functional.one_hot(torch.arange(0,dt.SUITS.HIGH))
        self.one_hot_ranks = torch.nn.functional.one_hot(torch.arange(0,dt.RANKS.HIGH))

        # Input is (b,4,2) -> (b,4,4) and (b,4,13)
        self.suit_conv = nn.Sequential(
            nn.Conv1d(5, 64, kernel_size=1, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.rank_conv = nn.Sequential(
            nn.Conv1d(5, 64, kernel_size=5, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.hidden_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i],hidden_dims[i+1]))
            self.bn_layers.append(nn.BatchNorm1d(64))
        self.dropout = nn.Dropout(0.5)
        self.categorical_output = nn.Linear(2048,self.nA)

    def forward(self,x):
        # Input is (b,5,2)
        M,c,h = x.size()
        ranks = x[:,:,0].long()
        suits = x[:,:,1].long()

        hot_ranks = self.one_hot_ranks[ranks]
        hot_suits = self.one_hot_suits[suits]

        s = self.suit_conv(hot_suits.float())
        r = self.rank_conv(hot_ranks.float())
        x = torch.cat((r,s),dim=-1)
        # should be (b,64,88)

        for i,hidden_layer in enumerate(self.hidden_layers):
            x = self.activation_fc(self.bn_layers[i](hidden_layer(x)))
        x = x.view(M,-1)
        x = self.dropout(x)
        return self.categorical_output(x)

################################################
#            Partial hand regression           #
################################################

class PartialHandRegression(nn.Module):
    def __init__(self,params,hidden_dims=(15,32,32),activation_fc=F.relu):
        super().__init__()
        self.params = params
        self.nA = params['nA']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.activation_fc = activation_fc
        self.seed = torch.manual_seed(params['seed'])
        
        self.one_hot_suits = torch.nn.functional.one_hot(torch.arange(0,dt.SUITS.HIGH))
        self.one_hot_ranks = torch.nn.functional.one_hot(torch.arange(0,dt.RANKS.HIGH))

        # Input is (b,4,2) -> (b,4,4) and (b,4,13)
        self.suit_conv = nn.Sequential(
            nn.Conv1d(9, 64, kernel_size=1, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.rank_conv = nn.Sequential(
            nn.Conv1d(9, 64, kernel_size=5, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.hidden_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i],hidden_dims[i+1]))
            self.bn_layers.append(nn.BatchNorm1d(64))
        self.dropout = nn.Dropout(0.5)
        self.categorical_output = nn.Linear(2048,self.nA)

    def forward(self,x):
        # Input is (b,5,2)
        M,c,h = x.size()
        ranks = x[:,:,0].long()
        suits = x[:,:,1].long()

        hot_ranks = self.one_hot_ranks[ranks]
        hot_suits = self.one_hot_suits[suits]

        s = self.suit_conv(hot_suits.float())
        r = self.rank_conv(hot_ranks.float())
        x = torch.cat((r,s),dim=-1)
        # should be (b,64,88)

        for i,hidden_layer in enumerate(self.hidden_layers):
            x = self.activation_fc(self.bn_layers[i](hidden_layer(x)))
        x = x.view(M,-1)
        x = self.dropout(x)
        return self.categorical_output(x)