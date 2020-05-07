
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

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
    action_probs =  action_masked / action_masked.sum(-1).unsqueeze(1)
    return action_probs

def combined_masks(action_mask,betsize_mask):
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
        actions = torch.zeros(self.nA)
        betsizes = torch.zeros(self.nB)
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

class WideAttention(nn.Module):
    def __init__(self,n_heads,d_model):
        super(WideAttention,self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.key = nn.Linear(d_model,d_model*n_heads,bias=False)
        self.query = nn.Linear(d_model,d_model*n_heads,bias=False)
        self.value = nn.Linear(d_model,d_model*n_heads,bias=False)
        self.unify_heads = nn.Linear(d_model*n_heads,d_model)
        
    def forward(self,x):
        b,t,k = x.size()
        h = self.n_heads
        query = self.query(x).view(b,t,h,k)
        key = self.key(x).view(b,t,h,k)
        value = self.value(x).view(b,t,h,k)
        
        query = query.transpose(1,2).contiguous().view(b*h,t,k)
        key = key.transpose(1,2).contiguous().view(b*h,t,k)
        value = value.transpose(1,2).contiguous().view(b*h,t,k)
        
        query = query / (k**1/4)
        key = key / (k**1/4)
        
        dot = torch.bmm(query,key.transpose(-2,-1))
        dot = F.softmax(dot,dim=-1)
        out = torch.bmm(dot,value).view(b,h,t,k)
        out = out.transpose(1,2).contiguous().view(b,t,h*k)
        return self.unify_heads(out)

class SelfAttention(nn.Module):
    def __init__(self, k, heads=8):
        super().__init__()
        self.k, self.heads = k, heads
        # These compute the queries, keys and values for all 
        # heads (as a single concatenated vector)
        self.tokeys    = nn.Linear(k, k * heads, bias=False)
        self.toqueries = nn.Linear(k, k * heads, bias=False)
        self.tovalues  = nn.Linear(k, k * heads, bias=False)

        # This unifies the outputs of the different heads into 
        # a single k-vector
        self.unifyheads = nn.Linear(heads * k, k)

    def forward(self, x):
        b, t, k = x.size()
        h = self.heads

        queries = self.toqueries(x).view(b, t, h, k)
        keys    = self.tokeys(x)   .view(b, t, h, k)
        values  = self.tovalues(x) .view(b, t, h, k)

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, k)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, k)
        values = values.transpose(1, 2).contiguous().view(b * h, t)
        queries = queries / (k ** (1/4))
        keys    = keys / (k ** (1/4))

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))
        # - dot has size (b*h, t, t) containing raw weights

        dot = F.softmax(dot, dim=2) 
        # - dot now contains row-wise normalized weights
        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, k)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, h * k)
        return self.unifyheads(out)

class TransformerBlock(nn.Module):
    def __init__(self, k, heads):
        super().__init__()
        
        self.attention = SelfAttention(k,heads=heads)
        
        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)
        
        self.ff = nn.Sequential(
            nn.Linear(k,4*k),
            nn.ReLU,
            nn.Linear(4*k,k)
        )
        
    def forward(self,x):
        attended = self.attention(x)
        x = self.norm1(attended + x)
        
        feedforward = self.ff(x)
        return self.norm2(feedforward + x)

################################################
#               Holdem Networks                #
################################################

class HoldemBaseline(nn.Module):
    def __init__(self,seed,nS,nA,params,hidden_dims=(64,64),activation=F.leaky_relu):
        super().__init__()
        self.activation = activation
        self.nS = nS
        self.nA = nA
        
        self.seed = torch.manual_seed(seed)
        self.mapping = params['mapping']
        self.action_emb = Embedder(6,64)

        # Input is (1,13,2) -> (1,13,64)
        self.one_hot_suits = torch.nn.functional.one_hot(torch.arange(0,dt.SUITS.HIGH))
        self.one_hot_ranks = torch.nn.functional.one_hot(torch.arange(0,dt.RANKS.HIGH))
        # Input is (b,4,2) -> (b,4,4) and (b,4,13)
        self.suit_conv = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=1, stride=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )
        self.rank_conv = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=5, stride=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )

        self.fc1 = nn.Linear(304,hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0],hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1],nA)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self,x,mask):
        M,c = x.size()
        ranks = x[:,self.mapping['state']['rank']].long()
        suits = x[:,self.mapping['state']['suit']].long()
        board_ranks = x[:,self.mapping['state']['board_ranks']].long()
        board_suits = x[:,self.mapping['state']['board_suits']].long()
        rank_input = torch.cat((ranks,board_ranks),dim=-1)
        suit_input = torch.cat((suits,board_suits),dim=-1)
        hot_ranks = self.one_hot_ranks[ranks]
        hot_suits = self.one_hot_suits[suits]

        s = self.suit_conv(hot_suits.float())
        r = self.rank_conv(hot_ranks.float())
        h = torch.cat((r,s),dim=-1)
        # should be (b,64,88)

        last_action = x[:,self.mapping['state']['previous_action']].long()
        last_action = self.action_emb(last_action)
        x = torch.cat([h.view(M,-1),last_action],dim=-1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        action_logits = self.fc3(x)
        
        action_probs = F.softmax(action_logits,dim=-1)
        action_probs = action_probs * mask
        action_probs /= action_probs.sum()
        m = Categorical(action_probs)
        action = m.sample()

        outputs = {
            'action':action,
            'action_prob':m.log_prob(action),
            'action_probs':action_probs
        }
        return outputs

class HoldemBaselineCritic(nn.Module):
    def __init__(self,seed,nO,nA,params,hidden_dims=(64,64),activation=F.leaky_relu):
        super().__init__()
        self.activation = activation
        self.nO = nO
        self.nA = nA
        
        self.seed = torch.manual_seed(seed)
        self.mapping = params['mapping']
        self.action_emb = Embedder(6,64)

        # Input is (1,13,2) -> (1,13,64)
        self.one_hot_suits = torch.nn.functional.one_hot(torch.arange(0,dt.SUITS.HIGH))
        self.one_hot_ranks = torch.nn.functional.one_hot(torch.arange(0,dt.RANKS.HIGH))
        # Input is (b,4,2) -> (b,4,4) and (b,4,13)
        self.suit_conv = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=1, stride=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )
        self.rank_conv = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=5, stride=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )

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
    def __init__(self,seed,nO,nA,params,hidden_dims=(64,64),activation=F.leaky_relu):
        super().__init__()
        self.activation = activation
        self.nO = nO
        self.nA = nA
        
        self.seed = torch.manual_seed(seed)
        self.mapping = params['mapping']
        self.action_emb = Embedder(6,64)

        # Input is (1,13,2) -> (1,13,64)
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

        self.fc1 = nn.Linear(304,hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0],hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1],nA)
        self.dropout = nn.Dropout(0.5)
        self.value_output = nn.Linear(5,1)
        self.advantage_output = nn.Linear(5,self.nA)

    def forward(self,x):
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
        x = self.activation(self.fc3(x))
        x = self.dropout(x)
        a = self.advantage_output(x)
        v = self.value_output(x)
        v = v.expand_as(a)
        q = v + a - a.mean(1,keepdim=True).expand_as(a)
        outputs = {
            'value':q
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
        hand = x[0,self.mapping['state']['rank']].long().unsqueeze(0)
        last_action = x[:,self.mapping['state']['previous_action']].long()
        previous_betsize = x[:,self.mapping['state']['previous_betsize']].float()
        if previous_betsize.dim() == 1:
            previous_betsize = previous_betsize.unsqueeze(1)
        hand = self.hand_emb(hand)
        last_action_emb = self.action_emb(last_action)
        x = torch.cat([hand,last_action_emb,previous_betsize],dim=-1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        cateogry_logits = self.fc3(x)
        cateogry_logits = self.noise(cateogry_logits)
        action_soft = F.softmax(cateogry_logits,dim=-1)
        action_probs = norm_frequencies(action_soft,mask)
        # action_probs = action_probs * mask
        # action_probs /= torch.sum(action_probs)
        m = Categorical(action_probs)
        action = m.sample()

        action_category,betsize_category = self.helper_functions.unwrap_action(action,last_action)
        
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