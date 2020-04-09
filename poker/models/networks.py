
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
#         print('action_probs',action_probs)
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


class CardClassification(nn.Module):
    def __init__(self,seed,state_space,hidden_dims=(64,32),activation_fc=F.relu):
        super(CardClassification,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.activation_fc = activation_fc
        self.seed = torch.manual_seed(seed)
        # Input is (1,13,2) -> (1,13,64)
        self.conv1 = nn.Conv1d(13, 64, kernel_size=2, stride=1)
        self.bn1 = nn.BatchNorm1d(64)
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
        # print(x.size())
        x = self.activation_fc(self.bn1(self.conv1(x)))
        # x = self.activation_fc(self.bn2(self.conv2(x)))
        # x = self.activation_fc(self.bn3(self.conv3(x)))
        # Flatten layer but retain number of samples
        # print(x.size())
        x = x.view(x.shape[0],x.shape[1])# * x.shape[2] * x.shape[3] * x.shape[4])
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        v = torch.tanh(self.value_output(x))
        # print(v.size())
        return v
        