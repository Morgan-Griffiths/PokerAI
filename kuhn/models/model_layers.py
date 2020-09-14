import torch.nn as nn
import torch.nn.functional as F
import torch
from models.model_utils import strip_padding

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

class positionalEncoder(nn.Module):
    def __init__(self,d_model, max_seq_len = 80):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(maxlen_seq,d_model)
        for pos in range(maxlen):
            for i in range(0,d_model,2):
                pe[pos,i] = pos / math.sin(10000 ** ((2*i)/d_model))
                pe[pos,i+1] = pos / math.cos(10000 ** ((2*(i+1))/d_model))
        pe.unsqueeze(0)
        self.register_buffer('pe',pe)
        
    def forward(self,x):
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        x = x + Variable(pe[:,:seq_len],requires_grad=False).cuda()
        return x

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
        int_actions = torch.argmax(actions, dim=0).unsqueeze(-1)
        int_betsizes = torch.argmax(betsizes, dim=0).unsqueeze(-1)
        return int_actions,int_betsizes

    # def unwrap_action(self,action:torch.Tensor,previous_action:torch.Tensor):
    #     """Unwraps flat action into action_category and betsize_category"""
    #     # print(action,previous_action)
    #     actions_output = torch.zeros(action.size(0),self.nA)
    #     betsizes = torch.zeros(action.size(0),self.nB)
    #     # actions[action[action < 3]] = 1
    #     # for i,action in enumerate(actions):
    #     if action < 3:
    #         actions_output[:,action] = 1
    #     elif previous_action == 5 or previous_action == 0: # Unopened
    #         actions_output[:,3] = 1
    #         bet_category = action - 3
    #         betsizes[:,bet_category] = 1
    #     else: # facing bet or raise
    #         actions_output[:,4] = 1
    #         bet_category = action - 3
    #         betsizes[:,bet_category] = 1
    #     int_actions = torch.argmax(actions_output, dim=-1)
    #     int_betsizes = torch.argmax(betsizes, dim=-1)
    #     return int_actions,int_betsizes

class PreProcessHistory(nn.Module):
    def __init__(self,params,critic=False):
        super().__init__()
        self.mapping = params['mapping']
        self.hand_emb = Embedder(5,255)
        self.action_emb = Embedder(6,256)
        self.betsize_fc = nn.Linear(1,256)
        self.maxlen = 10
        self.initialize(critic)

    def initialize(self,critic):
        if critic:
            # self.one_hot_kuhn = torch.nn.functional.one_hot(torch.arange(0,4))
            # self.one_hot_actions = torch.nn.functional.one_hot(torch.arange(0,6))
            # self.conv = nn.Sequential(
            #     nn.Conv1d(2, 32, kernel_size=3, stride=1),
            #     nn.BatchNorm1d(32),
            #     nn.ReLU(inplace=True)
            # )
            self.forward = self.forward_critic
        else:
            self.forward = self.forward_actor

    def forward_critic(self,x):
        stripped_x = strip_padding(x,self.maxlen).squeeze(0)
        M,C = stripped_x.size()
        hand = stripped_x[:,self.mapping['state']['rank']].long()
        h = self.hand_emb(hand)
        last_action = stripped_x[:,self.mapping['state']['previous_action']].long()
        last_action_emb = self.action_emb(last_action)
        # o.size(B,M,5)
        last_betsize = stripped_x[:,self.mapping['state']['previous_betsize']].float()
        if last_betsize.dim() == 1:
            last_betsize = last_betsize.unsqueeze(1)
        # h.size(B,M,128)
        combined = torch.cat([h,last_action_emb,last_betsize],dim=-1)
        return combined

    def forward_actor(self,x):
        stripped_x = strip_padding(x,self.maxlen).squeeze(0)
        hand = stripped_x[:,self.mapping['state']['rank']].long()
        hand = self.hand_emb(hand)
        # h.size(B,M,240)
        last_action = stripped_x[:,self.mapping['state']['previous_action']].long()
        last_action_emb = self.action_emb(last_action)
        # o.size(B,M,5)
        previous_betsize = stripped_x[:,self.mapping['state']['previous_betsize']].float()
        if previous_betsize.dim() == 1:
            previous_betsize = previous_betsize.unsqueeze(1)
        # h.size(B,M,128)
        # b1 = self.betsize_fc(previous_betsize)
        combined = torch.cat([hand,last_action_emb,previous_betsize],dim=-1)
        return combined


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

        return x #F.log_softmax(x, dim=1)
