
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from models.model_utils import padding_index
from models.buffers import PriorityReplayBuffer,PriorityTree
import hand_recognition.datatypes as dt

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
        self.device = params['device']
        self.process_input = PreProcessLayer(params)
        
        # self.seed = torch.manual_seed(seed)
        self.state_mapping = params['state_mapping']
        self.noise = GaussianNoise(self.device)
        self.emb = 512
        n_heads = 128
        depth = 8
        # self.transformer = CTransformer(self.emb,n_heads,depth,self.maxlen,self.nA)
        self.lstm = nn.LSTM(self.emb,128)
        self.action_out = nn.Linear(1280,self.nA)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self,state,action_mask,betsize_mask):
        x = state
        if not isinstance(x,torch.Tensor):
            x = torch.tensor(x,dtype=torch.float32).to(self.device)
            action_mask = torch.tensor(action_mask,dtype=torch.long).to(self.device)
            betsize_mask = torch.tensor(betsize_mask,dtype=torch.long).to(self.device)
        mask = combined_masks(action_mask,betsize_mask)
        out = self.process_input(x.permute(1,0,2))
        M,B,c = out.size()
        n_padding = self.maxlen - M
        if n_padding <= 0:
            h = out[-self.maxlen:,:,:]
        else:
            padding = torch.zeros(n_padding,B,out.size(-1)).to(self.device)
            h = torch.cat((out,padding),dim=0)
        # t_logits = self.transformer(out.permute(1,0,2))
        # lstm_out,_ = self.lstm(h)
        # lstm_out = lstm_out.permute(1,0,2)
        # print(lstm_out.size())
        t_logits = self.action_out(h.permute(1,0,2).contiguous().view(B,-1))
        category_logits = self.noise(t_logits)
        # print(category_logits.size())
        
        action_soft = F.softmax(category_logits,dim=-1)
        action_probs = norm_frequencies(action_soft,mask)
        m = Categorical(action_probs)
        action = m.sample()
        action_prob = m.log_prob(action)
        if B == 1:
            action_category,betsize_category = self.helper_functions.unwrap_action(action,state[:,-1,self.state_mapping['last_action']])
            action = action.item()
            action_category = action_category.item()
            betsize_category = betsize_category.item()
        else:
            action_category = None
            betsize_category = None
        outputs = {
            'action':action,
            'action_category':action_category,
            'action_prob':action_prob,
            'action_probs':action_probs,
            'betsize':betsize_category
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
        self.device = params['device']
        self.mapping = params['state_mapping']
        self.emb = 128
        n_heads = 8
        depth = 2
        self.transformer = CTransformer(self.emb,n_heads,depth,self.maxlen,128)
        self.dropout = nn.Dropout(0.5)
        self.value_output = nn.Linear(128,1)
        self.advantage_output = nn.Linear(128,self.combined_output)

    def forward(self,state):
        x = state
        if not isinstance(x,torch.Tensor):
            x = torch.tensor(x,dtype=torch.float32).to(self.device)
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
        # if action > 2:
        # generate betsize
        b = self.activation(self.bfc1(x))
        b = self.activation(self.bfc2(b))
        b = self.bfc3(b)
        betsize_logits = self.noise(b)
        betsize_probs = F.softmax(betsize_logits,dim=-1)
        if betsize_mask.sum(-1) == 0:
            betsize_mask = torch.ones(M,self.nA)
        # with torch.no_grad():
        mask_betsize_probs = betsize_probs * betsize_mask
        norm_betsize_probs = mask_betsize_probs / mask_betsize_probs.sum(-1).unsqueeze(1)
        b = Categorical(norm_betsize_probs)
        betsize = b.sample()
        betsize_prob = b.log_prob(betsize)
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

class FlatAC(nn.Module):
    def __init__(self,seed,nS,nA,nB,params,hidden_dims=(256,128),activation=F.leaky_relu):
        """
        Network capable of processing any number of prior actions
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
        self.preprocess = PreProcessHistory(params)
        self.max_length = 10
        emb = 128
        n_heads = 8
        depth = 2
        self.positional_emb = Embedder(self.max_length,128)
        self.transformer = CTransformer(emb,n_heads,depth,self.max_length,self.combined_output,max_pool=False)
        self.seed = torch.manual_seed(seed)
        self.mapping = params['mapping']
        self.noise = GaussianNoise(is_relative_detach=True)
        self.fc1 = nn.Linear(513,hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0],hidden_dims[1])
        self.fc3 = nn.Linear(1280,self.combined_output)
        self.value_output = nn.Linear(64,1)
        self.advantage_output = nn.Linear(64,self.combined_output)
        
    def forward(self,state,action_mask,betsize_mask):
        # last_state = state[-1].unsqueeze(0)
        mask = combined_masks(action_mask,betsize_mask)
        if mask.dim() > 1:
            mask = mask[-1]
        x = state
        M,C = x.size()
        out = self.preprocess(x)
        x = self.activation(self.fc1(out))
        x = self.activation(self.fc2(x))
        n_padding = self.max_length - M
        padding = torch.zeros(n_padding,out.size(-1))
        h = torch.cat((out,padding),dim=0)
        pos_emd = self.positional_emb(torch.arange(self.max_length))
        h = h + pos_emd
        # x = (h + pos_emd).unsqueeze(0)
        t_logits = self.fc3(h.view(-1)).unsqueeze(0)
        # t_logits = self.transformer(x)
        cateogry_logits = self.noise(t_logits)
        # distribution_inputs = F.log_softmax(cateogry_logits, dim=1) * mask
        action_soft = F.softmax(cateogry_logits,dim=-1)
        action_probs = norm_frequencies(action_soft,mask)
        last_action = state[-1,self.mapping['state']['previous_action']].long().unsqueeze(-1)
        m = Categorical(action_probs)
        action = m.sample()
        action_category,betsize_category = self.helper_functions.unwrap_action(action,last_action)
        
        q_input = x.view(M,-1)
        a = self.advantage_output(q_input)
        v = self.value_output(q_input)
        v = v.expand_as(a)
        q = v + a - a.mean(1,keepdim=True).expand_as(a)

        outputs = {
            'action':action,
            'action_category':action_category,
            'action_prob':m.log_prob(action),
            'action_probs':m.probs,
            'betsize':betsize_category,
            'value':q
            }
        return outputs

class FlatHistoricalActor(nn.Module):
    def __init__(self,seed,nS,nA,nB,params,hidden_dims=(64,64),activation=F.leaky_relu):
        """
        Network capable of processing any number of prior actions
        Num Categories: nA (check,fold,call,bet,raise)
        Num Betsizes: nB (various betsizes)
        """
        super().__init__()
        self.activation = activation
        # self.seed = torch.manual_seed(seed)
        self.nS = nS
        self.nA = nA
        self.nB = nB

        self.hand_emb = Embedder(5,64)
        self.action_emb = Embedder(6,63)
        self.combined_output = nA - 2 + nB
        self.helper_functions = NetworkFunctions(self.nA,self.nB)
        self.preprocess = PreProcessHistory(params)
        self.max_length = 10
        self.emb = 512
        n_heads = 8
        depth = 2
        self.positional_emb = Embedder(self.max_length,128)
        self.lstm = nn.LSTM(self.emb, 256)
        # self.transformer = CTransformer(self.emb,n_heads,depth,self.max_length,self.combined_output,max_pool=False)
        self.mapping = params['mapping']
        self.noise = GaussianNoise(is_relative_detach=True)
        self.fc1 = nn.Linear(128,hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0],hidden_dims[1])
        self.fc3 = nn.Linear(2560,self.combined_output)
        
    def forward(self,state,action_mask,betsize_mask):
        mask = combined_masks(action_mask,betsize_mask)
        if mask.dim() > 1:
            mask = mask[-1]
        x = state
        if x.dim() == 2:
            x = x.unsqueeze(0)
        out = self.preprocess(x)
        M,C = out.size()
        n_padding = self.max_length - M
        padding = torch.zeros(n_padding,out.size(-1))
        h = torch.cat((out,padding),dim=0).unsqueeze(0)
        # pos_emd = self.positional_emb(torch.arange(self.max_length))
        # padding_mask_o = torch.ones(M,self.emb)
        # padding_mask_z = torch.zeros(n_padding,self.emb)
        # padding_mask = torch.cat((padding_mask_o,padding_mask_z),dim=0)
        # pos_emd = (pos_emd.view(-1) * padding_mask.view(-1)).view(h.size(0),self.emb)
        # h = h + pos_emd
        # x = (h + pos_emd).unsqueeze(0)
        # x = self.activation(self.fc1(h))
        # x = self.activation(self.fc2(x)).view(-1)
        # t_logits = self.fc3(x).unsqueeze(0)
        x,_ = self.lstm(h)
        # x_stripped = (x.view(-1) * padding_mask.view(-1)).view(1,-1)
        t_logits = self.fc3(x.view(-1))
        # t_logits = self.transformer(x)
        cateogry_logits = self.noise(t_logits)
        # distribution_inputs = F.log_softmax(cateogry_logits, dim=1) * mask
        action_soft = F.softmax(cateogry_logits,dim=-1)
        action_probs = norm_frequencies(action_soft,mask)
        last_action = state[M-1,self.mapping['state']['previous_action']].long().unsqueeze(-1)
        m = Categorical(action_probs)
        action = m.sample()
        action_category,betsize_category = self.helper_functions.unwrap_action(action,last_action)
        
        outputs = {
            'action':action,
            'action_category':action_category,
            'action_prob':m.log_prob(action),
            'action_probs':m.probs,
            'betsize':betsize_category
            }
        return outputs

class FlatHistoricalCritic(nn.Module):
    def __init__(self,seed,nS,nA,nB,params,hidden_dims=(64,64),activation=F.leaky_relu):
        super().__init__()
        self.activation = activation
        # self.seed = torch.manual_seed(seed)
        self.nS = nS
        self.nA = nA
        self.nB = nB
        self.combined_output = nA - 2 + nB
        self.max_length = 10
        self.emb = 512
        n_heads = 8
        depth = 4
        nA = 128
        self.transformer = CTransformer(self.emb,n_heads,depth,self.max_length,nA)
        self.preprocess = PreProcessHistory(params,critic=True)
        self.value_output = nn.Linear(128,1)
        self.advantage_output = nn.Linear(128,self.combined_output)
        
    def forward(self,state):
        x = state
        if x.ndim == 2:
            x = x.unsqueeze(0)
        x = self.preprocess(x).unsqueeze(0)
        B,M,C = x.size()
        q_input = self.transformer(x)
        a = self.advantage_output(q_input)
        v = self.value_output(q_input)
        v = v.expand_as(a)
        q = v + a - a.mean(1,keepdim=True).expand_as(a)

        outputs = {'value':q }
        return outputs

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
        x = torch.cat([hand,last_action_emb,previous_betsize],dim=-1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        cateogry_logits = self.fc3(x)
        cateogry_logits = self.noise(cateogry_logits)
        action_soft = F.softmax(cateogry_logits,dim=-1)
        action_probs = norm_frequencies(action_soft,mask)
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
        
        self.use_embedding = params['embedding']
        self.mapping = params['mapping']
        self.one_hot_kuhn = torch.nn.functional.one_hot(torch.arange(0,4))
        self.one_hot_actions = torch.nn.functional.one_hot(torch.arange(0,6))
        self.hand_emb = Embedder(5,32)
        self.action_emb = Embedder(6,32)
        self.positional_embeddings = Embedder(2,32)
        self.fc0 = nn.Linear(64,hidden_dims[0])
        self.fc1 = nn.Linear(65,hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0],hidden_dims[1])
        self.value_output = nn.Linear(64,1)
        self.advantage_output = nn.Linear(64,self.combined_output)
        
    def forward(self,obs):
        x = obs
        M,c = x.size()
        hand = x[0,self.mapping['state']['rank']].long().unsqueeze(0)
        emb_hand = self.hand_emb(hand)
        last_action = x[:,self.mapping['state']['previous_action']].long()
        last_betsize = x[:,self.mapping['state']['previous_betsize']].float()
        if last_betsize.dim() == 1:
            last_betsize = last_betsize.unsqueeze(1)
        a1 = self.action_emb(last_action)

        h = emb_hand.view(-1).unsqueeze(0).repeat(M,1)
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
        
        # self.seed = torch.manual_seed(seed)
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
            x = torch.tensor(x,dtype=torch.float32)
            x = x.unsqueeze(0)
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
        
        # self.seed = torch.manual_seed(seed)
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
        
        # self.seed = torch.manual_seed(seed)
        self.use_embedding = params['embedding']
        self.mapping = params['mapping']
        self.one_hot_kuhn = torch.nn.functional.one_hot(torch.arange(0,4))
        self.one_hot_actions = torch.nn.functional.one_hot(torch.arange(0,6))
        self.hand_emb = Embedder(5,32)
        self.action_emb = Embedder(6,32)
        self.positional_embeddings = Embedder(2,32)

        # self.conv = nn.Sequential(
        #     nn.Conv1d(2, 32, kernel_size=3, stride=1),
        #     nn.BatchNorm1d(32),
        #     nn.ReLU(inplace=True)
        # )
        self.fc0 = nn.Linear(64,hidden_dims[0])
        self.fc1 = nn.Linear(64,hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0],hidden_dims[1])
        self.value_output = nn.Linear(64,1)
        self.advantage_output = nn.Linear(64,self.nC)
        
    def forward(self,state):
        x = state
        M,c = x.size()
        hand = x[:,self.mapping['state']['rank']].long()
        emb_hand = self.hand_emb(hand)
        # vil_hand = x[:,self.mapping['observation']['vil_rank']].long()
        # hands = torch.stack((hand,vil_hand)).permute(1,0)
        last_action = x[:,self.mapping['state']['previous_action']].long()
        a1 = self.action_emb(last_action)
        # hot_ranks = self.one_hot_kuhn[hands.long()]
        # if hot_ranks.dim() == 2:
        #     hot_ranks = hot_ranks.unsqueeze(0)
        # h = self.conv(hot_ranks.float()).view(M,-1)
        x = torch.cat([emb_hand,a1],dim=-1)
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