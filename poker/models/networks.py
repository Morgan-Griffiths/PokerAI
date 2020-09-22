import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from poker_env.datatypes import Action

from models.model_utils import padding_index
from models.buffers import PriorityReplayBuffer,PriorityTree

from models.model_layers import Embedder,GaussianNoise,PreProcessHistory,PreProcessPokerInputs,PreProcessLayer,CTransformer,NetworkFunctions
from models.model_utils import mask_,hard_update,combined_masks,norm_frequencies,strip_padding

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
        self.maxlen = params['maxlen']
        self.process_input = PreProcessPokerInputs(params)
        
        # self.seed = torch.manual_seed(seed)
        self.mapping = params['mapping']
        self.hand_emb = Embedder(5,64)
        self.action_emb = Embedder(6,64)
        self.betsize_emb = Embedder(self.nB,64)
        self.noise = GaussianNoise()
        self.emb = 1248
        n_heads = 8
        depth = 2
        self.lstm = nn.LSTM(self.emb, 128)
        # self.transformer = CTransformer(emb,n_heads,depth,self.max_length,self.nA)

        self.fc1 = nn.Linear(528,hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0],hidden_dims[1])
        self.fc3 = nn.Linear(1280,self.combined_output)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self,state,action_mask,betsize_mask):
        mask = combined_masks(action_mask,betsize_mask)
        x = state
        if x.dim() == 2:
            x = x.unsqueeze(0)
        out = self.process_input(x).unsqueeze(0)
        B,M,c = out.size()
        n_padding = max(self.maxlen - M,0)
        padding = torch.zeros(B,n_padding,out.size(-1))
        h = torch.cat((out,padding),dim=1)
        lstm_out,_ = self.lstm(h)
        t_logits = self.fc3(lstm_out.view(-1))
        category_logits = self.noise(t_logits)
        
        action_soft = F.softmax(category_logits,dim=-1)
        action_probs = norm_frequencies(action_soft,mask)
        m = Categorical(action_probs)
        action = m.sample()

        action_category,betsize_category = self.helper_functions.unwrap_action(action,state[:,-1,self.mapping['state']['previous_action']])
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
        
        # self.seed = torch.manual_seed(seed)
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
        
        self.process_input = PreProcessPokerInputs(params)
        self.maxlen = params['maxlen']
        self.mapping = params['mapping']
        emb = 1248
        n_heads = 8
        depth = 2
        self.transformer = CTransformer(emb,n_heads,depth,self.maxlen,self.nA)
        self.dropout = nn.Dropout(0.5)
        self.value_output = nn.Linear(5,1)
        self.advantage_output = nn.Linear(5,self.nA)

    def forward(self,state):
        x = state
        if x.ndim == 2:
            x = x.unsqueeze(0)
        out = self.process_input(x).unsqueeze(0)
        B,M,c = out.size()
        q_input = self.transformer(out)
        a = self.advantage_output(q_input)
        v = self.value_output(q_input)
        v = v.expand_as(a)
        q = v + a - a.mean(-1,keepdim=True).expand_as(a)
        outputs = {
            'value':q.squeeze(0)
            }
        return outputs

################################################
#                Omaha Networks                #
################################################

class OmahaActor(nn.Module):
    def __init__(self,seed,nS,nA,nB,params,hidden_dims=(64,64),activation=F.leaky_relu):
        super().__init__()
        self.activation = activation
        self.nS = nS
        self.nA = nA
        self.nB = nB
        self.combined_output = nA - 2 + nB
        self.helper_functions = NetworkFunctions(self.nA,self.nB)
        self.maxlen = params['maxlen']
        self.process_input = PreProcessLayer(params)
        
        # self.seed = torch.manual_seed(seed)
        self.state_mapping = params['state_mapping']
        self.hand_emb = Embedder(5,64)
        self.action_emb = Embedder(Action.UNOPENED,64)
        self.betsize_emb = Embedder(self.nB,64)
        self.noise = GaussianNoise()
        self.emb = 1248
        n_heads = 8
        depth = 2
        self.lstm = nn.LSTM(1280, 128)
        # self.transformer = CTransformer(emb,n_heads,depth,self.max_length,self.nA)

        self.fc1 = nn.Linear(528,hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0],hidden_dims[1])
        self.fc3 = nn.Linear(1280,self.combined_output)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self,state,action_mask,betsize_mask):
        x = torch.tensor(state,dtype=torch.float32)
        action_mask = torch.tensor(action_mask,dtype=torch.float)
        betsize_mask = torch.tensor(betsize_mask,dtype=torch.float)
        mask = combined_masks(action_mask,betsize_mask)

        out = self.process_input(x)
        B,M,c = out.size()
        n_padding = self.maxlen - M
        if n_padding < 0:
            h = out[:,-self.maxlen:,:]
        else:
            padding = torch.zeros(B,n_padding,out.size(-1))
            h = torch.cat((out,padding),dim=1)
        lstm_out,_ = self.lstm(h)
        t_logits = self.fc3(lstm_out.view(-1))
        category_logits = self.noise(t_logits)
        
        action_soft = F.softmax(category_logits,dim=-1)
        action_probs = norm_frequencies(action_soft,mask)
        m = Categorical(action_probs)
        action = m.sample()

        action_category,betsize_category = self.helper_functions.unwrap_action(action,state[:,-1,self.state_mapping['last_action']])
        outputs = {
            'action':action.item(),
            'action_category':action_category.item(),
            'action_prob':m.log_prob(action),
            'action_probs':action_probs,
            'betsize':betsize_category.item()
            }
        return outputs
    
class OmahaQCritic(nn.Module):
    def __init__(self,seed,nO,nA,nB,params,hidden_dims=(64,64),activation=F.leaky_relu):
        super().__init__()
        self.activation = activation
        self.nO = nO
        self.nA = nA
        self.combined_output = nA - 2 + nB
        self.process_input = PreProcessLayer(params)
        self.maxlen = params['maxlen']
        self.mapping = params['state_mapping']
        # self.emb = params['embedding_size']
        # self.lstm = nn.LSTM(1280, 128)
        emb = 1280
        n_heads = 8
        depth = 2
        self.transformer = CTransformer(emb,n_heads,depth,self.maxlen,128)
        self.dropout = nn.Dropout(0.5)
        self.value_output = nn.Linear(128,1)
        self.advantage_output = nn.Linear(128,self.combined_output)

    def forward(self,state):
        x = torch.tensor(state,dtype=torch.float32)
        out = self.process_input(x)
        # B,M,c = out.size()
        # n_padding = max(self.maxlen - M,0)
        # padding = torch.zeros(B,n_padding,out.size(-1))
        # h = torch.cat((out,padding),dim=1)

        q_input = self.transformer(out)
        a = self.advantage_output(q_input)
        v = self.value_output(q_input)
        v = v.expand_as(a)
        q = v + a - a.mean(-1,keepdim=True).expand_as(a)
        outputs = {
            'value':q.squeeze(0)
            }
        return outputs

class PredictionNet(nn.Module):
    def __init__(self,seed,nO,nA,nB,params,hidden_dims=(64,64),activation=F.leaky_relu):
        super().__init__()
        self.activation = activation
        self.nO = nO
        self.nA = nA
        self.combined_output = nA - 2 + nB
        self.process_input = PreProcessLayer(params)
        self.maxlen = params['maxlen']
        self.mapping = params['state_mapping']
        # self.emb = params['embedding_size']
        # self.lstm = nn.LSTM(1280, 128)
        emb = 2048
        n_heads = 8
        depth = 2
        self.transformer = CTransformer(emb,n_heads,depth,self.maxlen,128)
        self.dropout = nn.Dropout(0.5)
        self.value_output = nn.Linear(128,1)
        self.advantage_output = nn.Linear(128,self.combined_output)

    def forward(self,state):
        x = torch.tensor(state,dtype=torch.float32)
        out = self.process_input(x)
        
        # Actor
        B,M,c = out.size()
        n_padding = self.maxlen - M
        if n_padding < 0:
            h = out[:,-self.maxlen:,:]
        else:
            padding = torch.zeros(B,n_padding,out.size(-1))
            h = torch.cat((out,padding),dim=1)
        lstm_out,_ = self.lstm(h)
        t_logits = self.fc3(lstm_out.view(-1))
        category_logits = self.noise(t_logits)
        
        action_soft = F.softmax(category_logits,dim=-1)
        action_probs = norm_frequencies(action_soft,mask)
        m = Categorical(action_probs)
        action = m.sample()

        action_category,betsize_category = self.helper_functions.unwrap_action(action,state[:,-1,self.state_mapping['last_action']])
        outputs = {
            'action':action.item(),
            'action_category':action_category.item(),
            'action_prob':m.log_prob(action),
            'action_probs':action_probs,
            'betsize':betsize_category.item()
            }
        # Critic
        q_input = self.transformer(out)
        a = self.advantage_output(q_input)
        v = self.value_output(q_input)
        v = v.expand_as(a)
        q = v + a - a.mean(-1,keepdim=True).expand_as(a)
        outputs['value'] = q.squeeze(0)
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