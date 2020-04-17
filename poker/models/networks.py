
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from models.buffers import PriorityReplayBuffer,PriorityTree

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

def hard_update(source,target):
    for target_param,param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

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

class Baseline(nn.Module):
    def __init__(self,seed,nS,nA,params,hidden_dims=(64,64),activation=F.leaky_relu):
        super(Baseline,self).__init__()
        self.activation = activation
        self.nS = nS
        self.nA = nA
        
        self.seed = torch.manual_seed(seed)
        self.use_embedding = params['embedding']
        print(f'Using embeddings {self.use_embedding}')
        self.mapping = params['mapping']
        self.hand_emb = Embedder(5,64)
        self.action_emb = Embedder(6,64)
        if self.use_embedding == True:
            self.fc1 = nn.Linear(64+64,hidden_dims[0])
            self.fc2 = nn.Linear(hidden_dims[0],hidden_dims[1])
            self.fc3 = nn.Linear(hidden_dims[1],nA)
        else:
            self.fc1 = nn.Linear(nS,hidden_dims[0])
            self.fc2 = nn.Linear(hidden_dims[0],hidden_dims[1])
            self.fc3 = nn.Linear(hidden_dims[1],nA)
        
    def forward(self,state,mask):
        x = state
        if not isinstance(state,torch.Tensor):
            x = torch.tensor(x,dtype=torch.float32) #device = self.device,
            x = x.unsqueeze(0)
        if self.use_embedding:
            hand = x[:,self.mapping['hand']].long()
            last_action = x[:,self.mapping['action']].long()
            hand = self.hand_emb(hand)
            last_action = self.action_emb(last_action)
            x = torch.cat([hand,last_action],dim=-1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        action_logits = self.fc3(x)
        
        action_probs = F.softmax(action_logits,dim=-1)
        action_probs = action_probs * mask
        action_probs /= torch.sum(action_probs)
        m = Categorical(action_probs)
        action = m.sample()
        return action,m.log_prob(action)

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

# Conv + FC
class CardClassification(nn.Module):
    def __init__(self,params,hidden_dims=(64,32),activation_fc=F.relu):
        super(CardClassification,self).__init__()
        self.params = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.activation_fc = activation_fc
        self.seed = torch.manual_seed(params['seed'])
        # Input is (1,13,2) -> (1,13,64)
        self.hidden_conv_layers = nn.ModuleList()
        self.hidden_bn_layers = nn.ModuleList()
        for i in range(params['conv_layers']):
            self.conv = nn.Conv1d(params['channels'][i], 64, kernel_size=params['kernel'][i], stride=1)
            self.hidden_conv_layers.append(self.conv)
            if params['batchnorm'] == True:
                self.bn = nn.BatchNorm1d(64)
                self.hidden_bn_layers.append(self.bn)
            # Output shape is (1,64,9,4,4)
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i],hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        self.value_output = nn.Linear(hidden_dims[-1],1)

    def forward(self,state):
        x = state
        if not isinstance(state,torch.Tensor):
            x = torch.tensor(x,dtype=torch.float32,device = self.device)
            # x = x.unsqueeze(0)
        M = x.size(0)
        if self.params['permute'] == True:
            x = x.permute(0,2,1)
        for i,layer in enumerate(self.hidden_conv_layers):
            # print(x.size())
            if len(self.hidden_bn_layers):
                x = self.activation_fc(self.hidden_bn_layers[i](layer(x)))
            else:
                x = self.activation_fc(layer(x))
        # x = self.activation_fc(self.bn2(self.conv2(x)))
        # x = self.activation_fc(self.bn3(self.conv3(x)))
        # Flatten layer but retain number of samples
        x = x.view(M,-1) # * x.shape[2] * x.shape[3] * x.shape[4])
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        v = torch.tanh(self.value_output(x))
        return v
        
# Emb + fc
class CardClassificationV2(nn.Module):
    def __init__(self,params,hidden_dims=(44,64,572),activation_fc=F.relu):
        super(CardClassificationV2,self).__init__()
        self.params = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.activation_fc = activation_fc
        self.seed = torch.manual_seed(params['seed'])
        # Input is (1,13,2) -> (1,13,64)
        self.rank_emb = Embedder(15,32)
        self.suit_emb = Embedder(4,12)
        # Output shape is (1,64,9,4,4)
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i],hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        self.value_output = nn.Linear(hidden_dims[-1],1)

    def forward(self,state):
        x = state
        if not isinstance(state,torch.Tensor):
            x = torch.tensor(x,dtype=torch.float32,device = self.device)
            # x = x.unsqueeze(0)
        M = x.size(0)
        ranks = self.rank_emb(state[:,:,0].long())
        suits = self.suit_emb(state[:,:,1].long())
        embs = torch.cat((ranks,suits),dim=-1)
        x = embs
        # Flatten layer but retain number of samples
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        x = embs.view(M,-1) # * x.shape[2] * x.shape[3] * x.shape[4])
        v = torch.tanh(self.value_output(x))
        return v

# Conv + multiheaded attention
class CardClassificationV3(nn.Module):
    def __init__(self,params,hidden_dims=(64,32),activation_fc=F.relu):
        super(CardClassificationV3,self).__init__()
        self.params = params
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
        self.value_output = nn.Linear(hidden_dims[-1],1)

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

# Emb + fc
class HandClassification(nn.Module):
    def __init__(self,params,hidden_dims=(44,64,64),activation_fc=F.relu):
        super(HandClassification,self).__init__()
        self.params = params
        self.nA = params['nA']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.activation_fc = activation_fc
        self.seed = torch.manual_seed(params['seed'])
        
        self.rank_emb = Embedder(15,32)
        self.suit_emb = Embedder(4,12)
        self.public_rank_emb = Embedder(15,32)
        self.public_suit_emb = Embedder(4,12)
        
        self.hidden_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i],hidden_dims[i+1]))
            self.bn_layers.append(nn.BatchNorm1d(9))
        self.dropout = nn.Dropout(0.5)
        self.categorical_output = nn.Linear(576,self.nA)

    def forward(self,state):
        x = state
        if not isinstance(state,torch.Tensor):
            x = torch.tensor(x,dtype=torch.float32,device = self.device)
            # x = x.unsqueeze(0)
        M = x.size(0)
        hand = x[:,:4,:].long()
        ranks = self.rank_emb(hand[:,:,0])
        suits = self.suit_emb(hand[:,:,1])
        hand_embs = torch.cat((ranks,suits),dim=-1)
        board = x[:,4:,:].long()
        board_ranks = self.public_rank_emb(board[:,:,0])
        board_suits = self.public_suit_emb(board[:,:,1])
        board_embs = torch.cat((board_ranks,board_suits),dim=-1)
        x = torch.cat((hand_embs,board_embs),dim=1)

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
        # print(positions.size())
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
    def __init__(self,params,hidden_dims=(386,64,64),activation_fc=F.relu):
        super().__init__()
        self.params = params
        self.nA = params['nA']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.activation_fc = activation_fc
        self.seed = torch.manual_seed(params['seed'])

        # self.one_hot_suits = torch.nn.functional.one_hot(torch.arange(0,4))
        self.one_hot_ranks = torch.nn.functional.one_hot(torch.arange(0,15))

        # Input is (b,5,2) -> (b,5,13) or (b,5,15)
        self.conv1 = nn.Conv1d(5, 64, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm1d(64)
        # Output shape is (b,64,9) or (b,64,11) 
        self.rank_output = nn.Linear(11,6)
        self.flushfc = nn.Linear(5,2)
        self.hidden_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i],hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
            self.bn_layers.append(nn.BatchNorm2d(hidden_dims[i+1]))
        self.dropout = nn.Dropout(0.5)
        self.categorical_output = nn.Linear(hidden_dims[-1],self.nA)

    def forward(self,x):
        # Input is M,5,2
        assert(isinstance(x,torch.Tensor))
        M = x.size(0)
        ranks = x[:,:,0]
        suits = x[:,:,1]
        hot_ranks = self.one_hot_ranks[ranks.long()]
        r = self.activation_fc(self.bn1(self.conv1(hot_ranks.float())))
        r = self.activation_fc(self.rank_output(r))
        s = self.activation_fc(self.flushfc(suits))
        # hot_suits = self.one_hot_suits(suits)
        r = r.view(M,-1)
        x = torch.cat((s,r),dim=-1)
        for i,hidden_layer in enumerate(self.hidden_layers):
            # x = self.activation_fc(hidden_layer(x))
            x = self.activation_fc(self.bn_layers[i](hidden_layer(x)))
        x = self.dropout(x)
        return self.categorical_output(x)