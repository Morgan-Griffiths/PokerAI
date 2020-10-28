import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from poker_env.datatypes import Action

from models.model_utils import padding_index,count_parameters
from models.buffers import PriorityReplayBuffer,PriorityTree

from models.model_layers import EncoderAttention,VectorAttention,Embedder,GaussianNoise,PreProcessPokerInputs,PreProcessLayer,CTransformer,NetworkFunctions,IdentityBlock
from models.model_utils import mask_,hard_update,combined_masks,norm_frequencies,strip_padding

class BetAgent(object):
    def __init__(self):
        pass

    def name(self):
        return 'baseline_evaluation'

    def __call__(self,state,action_mask,betsize_mask):
        if betsize_mask.sum() > 0:
            action = np.argmax(betsize_mask,axis=-1) + 3
        else:
            action = np.argmax(action_mask,axis=-1)
        actor_outputs = {
            'action':action,
            'action_category':int(np.where(action_mask > 0)[-1][-1]),
            'action_probs':torch.zeros(5).fill_(2.),
            'action_prob':torch.tensor([1.]),
            'betsize' : int(np.argmax(betsize_mask,axis=-1))
        }
        return actor_outputs

################################################
#               Holdem Networks                #
################################################

class Network(nn.Module):
    def __init__(self):
        super().__init__()
    @property
    def summary(self):
        count_parameters(self)

class HoldemBaseline(Network):
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

class HoldemBaselineCritic(Network):
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

class HoldemQCritic(Network):
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


class OmahaBatchActor(Network):
    def __init__(self,seed,nS,nA,nB,params,hidden_dims=(64,64),activation=F.leaky_relu):
        super().__init__()
        self.activation = activation
        self.nS = nS
        self.nA = nA
        self.nB = nB
        self.combined_output = nA - 2 + nB
        self.helper_functions = NetworkFunctions(self.nA,self.nB)
        self.maxlen = params['maxlen']
        self.device = params['device']
        self.process_input = PreProcessLayer(params)
        
        # self.seed = torch.manual_seed(seed)
        self.state_mapping = params['state_mapping']
        self.hand_emb = Embedder(5,64)
        self.action_emb = Embedder(Action.UNOPENED,64)
        self.betsize_emb = Embedder(self.nB,64)
        self.noise = GaussianNoise(self.device)
        self.emb = 1248
        n_heads = 8
        depth = 2
        self.lstm = nn.LSTM(1280, 128,bidirectional=True)
        self.batchnorm = nn.BatchNorm1d(self.maxlen)
        # self.blocks = nn.Sequential(
        #     IdentityBlock(hidden_dims=(2560,2560,512),activation=F.leaky_relu),
        #     IdentityBlock(hidden_dims=(512,512,256),activation=F.leaky_relu),
        # )
        self.fc_final = nn.Linear(2560,self.combined_output)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self,state,action_mask,betsize_mask):
        x = torch.tensor(state,dtype=torch.float32).to(self.device)
        action_mask = torch.tensor(action_mask,dtype=torch.float).to(self.device)
        betsize_mask = torch.tensor(betsize_mask,dtype=torch.float).to(self.device)
        mask = combined_masks(action_mask,betsize_mask)

        out = self.process_input(x)
        B,M,c = out.size()
        n_padding = self.maxlen - M
        if n_padding < 0:
            h = out[:,-self.maxlen:,:]
        else:
            padding = torch.zeros(B,n_padding,out.size(-1)).to(self.device)
            h = torch.cat((out,padding),dim=1)
        lstm_out,_ = self.lstm(h)
        norm = self.batchnorm(lstm_out)
        # blocks_out = self.blocks(lstm_out.view(-1))
        t_logits = self.fc_final(norm.view(-1))
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
        
class OmahaBatchObsQCritic(Network):
    def __init__(self,seed,nO,nA,nB,params,hidden_dims=(64,64),activation=F.leaky_relu):
        super().__init__()
        self.activation = activation
        self.nO = nO
        self.nA = nA
        self.combined_output = nA - 2 + nB
        self.process_input = PreProcessLayer(params,critic=True)
        self.maxlen = params['maxlen']
        self.mapping = params['state_mapping']
        self.device = params['device']
        # self.emb = params['embedding_size']
        # self.lstm = nn.LSTM(1280, 128)
        emb = params['transformer_in']
        n_heads = 8
        depth = 2
        self.transformer = CTransformer(emb,n_heads,depth,self.maxlen,params['transformer_out'])
        self.dropout = nn.Dropout(0.5)
        self.value_output = nn.Linear(params['transformer_out'],1)
        self.advantage_output = nn.Linear(params['transformer_out'],self.combined_output)

    def forward(self,obs):
        x = torch.tensor(obs,dtype=torch.float32).to(self.device)
        out = self.process_input(x)
        q_input = self.transformer(out)
        a = self.advantage_output(q_input)
        v = self.value_output(q_input)
        v = v.expand_as(a)
        q = v + a - a.mean(-1,keepdim=True).expand_as(a)
        outputs = {
            'value':q.squeeze(0)
            }
        return outputs

def copy_weights(net,path):
    if torch.cuda.is_available():
        layer_weights = torch.load(path)
    else:
        layer_weights = torch.load(path,map_location=torch.device('cpu'))
    for name, param in net.process_input.hand_board.named_parameters():
        if name in layer_weights:
            print('update_weights',name)
            param.data.copy_(layer_weights[name].data)
            param.requires_grad = False

class OmahaActor(Network):
    def __init__(self,seed,nS,nA,nB,params,hidden_dims=(64,64),activation=F.leaky_relu):
        super().__init__()
        self.activation = activation
        self.nS = nS
        self.nA = nA
        self.nB = nB
        self.combined_output = nA - 2 + nB
        self.helper_functions = NetworkFunctions(self.nA,self.nB)
        self.maxlen = params['maxlen']
        self.device = params['device']
        self.process_input = PreProcessLayer(params)
        
        # self.seed = torch.manual_seed(seed)
        self.state_mapping = params['state_mapping']
        self.action_emb = Embedder(Action.UNOPENED,64)
        self.betsize_emb = Embedder(self.nB,64)
        self.noise = GaussianNoise(self.device)
        self.emb = 1248
        n_heads = 8
        depth = 2
        self.attention = EncoderAttention(params['lstm_in'],params['lstm_out'])
        self.lstm = nn.LSTM(params['lstm_in'],params['lstm_out'],bidirectional=True)
        self.batchnorm = nn.BatchNorm1d(self.maxlen)
        # self.blocks = nn.Sequential(
        #     IdentityBlock(hidden_dims=(2560,2560,512),activation=F.leaky_relu),
        #     IdentityBlock(hidden_dims=(512,512,256),activation=F.leaky_relu),
        # )
        self.fc_final = nn.Linear(2560,self.combined_output)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self,state,action_mask,betsize_mask):
        """
        state: B,M,39
        """
        x = state
        if not isinstance(x,torch.Tensor):
            x = torch.tensor(x,dtype=torch.float32).to(self.device)
            action_mask = torch.tensor(action_mask,dtype=torch.float).to(self.device)
            betsize_mask = torch.tensor(betsize_mask,dtype=torch.float).to(self.device)
        mask = combined_masks(action_mask,betsize_mask)
        out = self.process_input(x)
        B,M,c = out.size()
        n_padding = self.maxlen - M
        if n_padding < 0:
            h = out[:,-self.maxlen:,:]
        else:
            padding = torch.zeros(B,n_padding,out.size(-1)).to(self.device)
            h = torch.cat((padding,out),dim=1)
        lstm_out,hidden_states = self.lstm(h)
        norm = self.batchnorm(lstm_out)
        # self.attention(out)
        # blocks_out = self.blocks(lstm_out.view(-1))
        t_logits = self.fc_final(norm.view(B,-1))
        category_logits = self.noise(t_logits)
        # skip connection
        # category_logits += h
        action_soft = F.softmax(category_logits,dim=-1)
        action_probs = norm_frequencies(action_soft,mask)
        m = Categorical(action_probs)
        action = m.sample()
        previous_action = torch.as_tensor(state[:,-1,self.state_mapping['last_action']]).to(self.device)
        action_category,betsize_category = self.helper_functions.batch_unwrap_action(action,previous_action)
        if B > 1:
            # batch training
            outputs = {
                'action':action,
                'action_category':action_category,
                'action_prob':m.log_prob(action),
                'action_probs':action_probs,
                'betsize':betsize_category
                }
        else:
            # playing hand
            outputs = {
                'action':action.item(),
                'action_category':action_category.item(),
                'action_prob':m.log_prob(action),
                'action_probs':action_probs,
                'betsize':betsize_category.item()
                }
        return outputs
    
class OmahaQCritic(Network):
    def __init__(self,seed,nO,nA,nB,params,hidden_dims=(64,64),activation=F.leaky_relu):
        super().__init__()
        self.activation = activation
        self.nO = nO
        self.nA = nA
        self.combined_output = nA - 2 + nB
        self.process_input = PreProcessLayer(params)
        self.maxlen = params['maxlen']
        self.mapping = params['state_mapping']
        self.device = params['device']
        # self.emb = params['embedding_size']
        # self.lstm = nn.LSTM(1280, 128)
        emb = params['transformer_in']
        n_heads = 8
        depth = 2
        self.transformer = CTransformer(emb,n_heads,depth,self.maxlen,params['transformer_out'])
        self.dropout = nn.Dropout(0.5)
        self.value_output = nn.Linear(params['transformer_out'],1)
        self.advantage_output = nn.Linear(params['transformer_out'],self.combined_output)

    def forward(self,state):
        x = torch.tensor(state,dtype=torch.float32).to(self.device)
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

class OmahaObsQCritic(Network):
    def __init__(self,seed,nO,nA,nB,params,hidden_dims=(64,64),activation=F.leaky_relu):
        super().__init__()
        self.activation = activation
        self.nO = nO
        self.nA = nA
        self.combined_output = nA - 2 + nB
        self.attention = VectorAttention(params['transformer_in'])
        self.process_input = PreProcessLayer(params,critic=True)
        self.maxlen = params['maxlen']
        self.mapping = params['state_mapping']
        self.device = params['device']
        # self.emb = params['embedding_size']
        # self.lstm = nn.LSTM(1280, 128)
        emb = params['transformer_in']
        n_heads = 8
        depth = 2
        self.transformer = CTransformer(emb,n_heads,depth,self.maxlen,params['transformer_out'])
        self.dropout = nn.Dropout(0.5)
        self.value_output = nn.Linear(params['transformer_out'],1)
        self.advantage_output = nn.Linear(params['transformer_out'],self.combined_output)

    def forward(self,obs):
        x = obs
        if not isinstance(x,torch.Tensor):
            x = torch.tensor(x,dtype=torch.float32).to(self.device)
        out = self.process_input(x)
        # context = self.attention(out)
        q_input = self.transformer(out)
        a = self.advantage_output(q_input)
        v = self.value_output(q_input)
        v = v.expand_as(a)
        q = v + a - a.mean(-1,keepdim=True).expand_as(a)
        outputs = {
            'value':q.squeeze(0)
            }
        return outputs

class CombinedNet(Network):
    def __init__(self,seed,nO,nA,nB,params,hidden_dims=(64,64),activation=F.leaky_relu):
        super().__init__()
        self.activation = activation
        self.nO = nO
        self.nA = nA
        self.nB = nB
        self.combined_output = nA - 2 + nB
        self.maxlen = params['maxlen']
        self.mapping = params['state_mapping']
        self.device = params['device']
        # self.emb = params['embedding_size']
        self.helper_functions = NetworkFunctions(self.nA,self.nB)
        self.process_input = PreProcessLayer(params)
        self.lstm = nn.LSTM(1280, 128)
        self.policy_out = nn.Linear(1280,self.combined_output)
        self.noise = GaussianNoise(self.device)
        emb = params['transformer_in']
        n_heads = 8
        depth = 2
        self.transformer = CTransformer(emb,n_heads,depth,self.maxlen,params['transformer_out'])
        self.dropout = nn.Dropout(0.5)
        self.value_output = nn.Linear(params['transformer_out'],1)
        self.advantage_output = nn.Linear(params['transformer_out'],self.combined_output)

    def forward(self,state,action_mask,betsize_mask):
        x = torch.tensor(state,dtype=torch.float32).to(self.device)
        action_mask = torch.tensor(action_mask,dtype=torch.float).to(self.device)
        betsize_mask = torch.tensor(betsize_mask,dtype=torch.float).to(self.device)
        mask = combined_masks(action_mask,betsize_mask)
        out = self.process_input(x)
        # Actor
        B,M,c = out.size()
        n_padding = self.maxlen - M
        if n_padding < 0:
            h = out[:,-self.maxlen:,:]
        else:
            padding = torch.zeros(B,n_padding,out.size(-1)).to(self.device)
            h = torch.cat((out,padding),dim=1)
        lstm_out,_ = self.lstm(h)
        t_logits = self.policy_out(lstm_out.view(-1))
        category_logits = self.noise(t_logits)
        
        action_soft = F.softmax(category_logits,dim=-1)
        action_probs = norm_frequencies(action_soft,mask)
        m = Categorical(action_probs)
        action = m.sample()

        action_category,betsize_category = self.helper_functions.unwrap_action(action,state[:,-1,self.mapping['last_action']])
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