import torch
import torch.nn as nn
import torch.nn.functional as F
import datatypes as dt
import numpy as np
import math
from itertools import combinations
from prettytable import PrettyTable
from card_utils import load_obj

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

UNSPOOL_INDEX = np.array([h + b for h in combinations(range(0,4), 2) for b in combinations(range(4,9), 3)])

def flat_unspool(X):
    """
    Takes a flat (B,M,18) tensor vector of alternating ranks and suits, 
    sorts them with hand and board, and returns (B,M,60,5) vector
    """
    # Size of (B,M,18)
    ranks = X[:,:,::2]
    suits = X[:,:,1::2]
    cards = (ranks-1) * suits
    hand = cards[:,:,:4]
    board = cards[:,:,4:]
    # sort hand and board
    hand_index = torch.argsort(hand)
    board_index = torch.argsort(board)
    hand_sorted = torch.gather(hand,-1,hand_index)
    board_sorted = torch.gather(board,-1,board_index)
    combined = torch.cat((hand_sorted,board_sorted),dim=-1).squeeze(-1).long()
    # B,5
    sequence = combined[:,UNSPOOL_INDEX]
    return sequence

def unspool(X):
    """
    Takes a flat (B,M,18) tensor vector of alternating ranks and suits, 
    sorts them with hand and board, and returns (B,M,60,5) vector
    """
    # Size of (B,M,18)
    ranks = X[:,:,::2]
    suits = X[:,:,1::2]
    hand_ranks = ranks[:,:,:4]
    hand_suits = suits[:,:,:4]
    board_ranks = ranks[:,:,4:]
    board_suits = suits[:,:,4:]
    # sort by suit
    hand_suit_index = torch.argsort(hand_suits)
    board_suit_index = torch.argsort(board_suits)
    hand_ranks = torch.gather(hand_ranks,-1,hand_suit_index)
    hand_suits = torch.gather(hand_suits,-1,hand_suit_index)
    board_ranks = torch.gather(board_ranks,-1,board_suit_index)
    board_suits = torch.gather(board_suits,-1,board_suit_index)
    # sort by rank
    hand_index = torch.argsort(hand_ranks)
    board_index = torch.argsort(board_ranks)
    ranks = torch.cat((torch.gather(hand_ranks,-1,hand_index),torch.gather(board_ranks,-1,board_index)),dim=-1).long()
    suits = torch.cat((torch.gather(hand_suits,-1,hand_index),torch.gather(board_suits,-1,board_index)),dim=-1).long()
    sequence_ranks = ranks[:,:,UNSPOOL_INDEX]
    sequence_suits = suits[:,:,UNSPOOL_INDEX]
    # sequence_suits = swap_batch_suits(sequence_suits)
    return sequence_ranks,sequence_suits

def copy_weights(network,path):
    if torch.cuda.is_available():
        layer_weights = torch.load(path)
    else:
        layer_weights = torch.load(path,map_location=torch.device('cpu'))
    for name, param in network.named_parameters():
        if name in layer_weights:
            print('copying weights',name)
            param.data.copy_(layer_weights[name].data)
            param.requires_grad = False

################################################
#           13 card winner prediction          #
################################################

class ThirteenCard(nn.Module):
    def __init__(self,params,hidden_dims=(16,32),activation_fc=F.relu):
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
    def __init__(self,params,hidden_dims=(1024,512,512),activation_fc=F.relu):
        super().__init__()
        self.params = params
        self.nA = params['nA']
        self.gpu1 = params['gpu1']
        self.gpu2 = params['gpu2']
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
            # self.bn_layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
        self.categorical_output = nn.Linear(512,self.nA)

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

        if torch.cuda.is_available():
            r = self.rank_conv(hero_board_ranks.float().cuda())
            s = self.suit_conv(hero_board_suits.float().cuda())
            r2 = self.rank_conv(vil_board_ranks.float().cuda())
            s2 = self.suit_conv(vil_board_suits.float().cuda())
        else:
            r = self.rank_conv(hero_board_ranks.float())
            s = self.suit_conv(hero_board_suits.float())
            r2 = self.rank_conv(vil_board_ranks.float())
            s2 = self.suit_conv(vil_board_suits.float())
        x1 = torch.cat((r,s),dim=-1).view(M,-1)
        x2 = torch.cat((r2,s2),dim=-1).view(M,-1)
        for i,hidden_layer in enumerate(self.hidden_layers):
            x1 = self.activation_fc(hidden_layer(x1))
        for i,hidden_layer in enumerate(self.hidden_layers):
            x2 = self.activation_fc(hidden_layer(x2))

        x = x1 - x2
        x = x.view(M,-1)
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
    def __init__(self,params,hidden_dims=(16,32,32),activation_fc=F.relu):
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

        if torch.cuda.is_available():
            s = self.suit_conv(hot_suits.float().cuda())
            r = self.rank_conv(hot_ranks.float().cuda())
        else:
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
        self.suit_emb = Embedder(5,32)
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
        self.suit_emb = Embedder(5,12)
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
        self.suit_emb = Embedder(5,12)
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
        self.suit_output = nn.Linear(5,12)
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
    def __init__(self,params,hidden_dims=(16,32,32),activation_fc=F.relu):
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

class HandRankClassificationNine(nn.Module):
    def __init__(self,params,hidden_dims=(16,32,32),activation_fc=F.relu):
        super().__init__()
        self.params = params
        self.nA = params['nA']
        self.activation_fc = activation_fc
        self.seed = torch.manual_seed(params['seed'])
        
        self.one_hot_suits = torch.nn.functional.one_hot(torch.arange(0,dt.SUITS.HIGH))
        self.one_hot_ranks = torch.nn.functional.one_hot(torch.arange(0,dt.RANKS.HIGH))

        # Input is (b,4,2) -> (b,4,4) and (b,4,13)
        self.suit_conv = nn.Sequential(
            nn.Conv1d(5, 16, kernel_size=1, stride=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )
        self.rank_conv = nn.Sequential(
            nn.Conv1d(5, 16, kernel_size=5, stride=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )
        self.hidden_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i],hidden_dims[i+1]))
            # self.bn_layers.append(nn.BatchNorm1d(64))
        self.categorical_output = nn.Linear(2048,self.nA)

    def forward(self,x):
        # Input is (b,9,2)
        B,c,h = x.size()
        ranks = x[:,:,0]
        suits = x[:,:,1]
        interleaved = torch.empty((B,c*h))
        interleaved[:,0::2] = ranks
        interleaved[:,1::2] = suits
        ranks,suits = unspool(interleaved)
        hot_ranks = self.one_hot_ranks[ranks].float().to(self.device)
        hot_suits = self.one_hot_suits[suits].float().to(self.device)
        # (b,60,5,2)
        for i in range(B):
            s = self.suit_conv(hot_suits)
            r = self.rank_conv(hot_ranks)
            x = torch.cat((r,s),dim=-1)
            # should be (b,64,88)
            for i,hidden_layer in enumerate(self.hidden_layers):
                x = self.activation_fc(hidden_layer(x))
            x = x.view(M,-1)
            self.categorical_output(x)
        return 

class HandRankClassificationFive(nn.Module):
    def __init__(self,params,hidden_dims=(16,32,32),activation_fc=F.relu):
        super().__init__()
        self.params = params
        self.nA = params['nA']
        self.activation_fc = activation_fc
        self.seed = torch.manual_seed(params['seed'])
        
        self.one_hot_suits = torch.nn.functional.one_hot(torch.arange(0,dt.SUITS.HIGH))
        self.one_hot_ranks = torch.nn.functional.one_hot(torch.arange(0,dt.RANKS.HIGH))

        # Input is (b,4,2) -> (b,4,4) and (b,4,13)
        self.suit_conv = nn.Sequential(
            nn.Conv1d(5, 16, kernel_size=1, stride=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )
        self.rank_conv = nn.Sequential(
            nn.Conv1d(5, 16, kernel_size=5, stride=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )
        self.hidden_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i],hidden_dims[i+1]))
            # self.bn_layers.append(nn.BatchNorm1d(64))
        self.categorical_output = nn.Linear(512,self.nA)

    def forward(self,x):
        # Input is (b,5,2)
        M,c,h = x.size()
        ranks = x[:,:,0].long()
        suits = x[:,:,1].long()
        hot_ranks = self.one_hot_ranks[ranks]
        hot_suits = self.one_hot_suits[suits]
        # hot_ranks is (b,5,15)
        # hot_ranks is (b,5,5)
        if torch.cuda.is_available():
            s = self.suit_conv(hot_suits.float().cuda())
            r = self.rank_conv(hot_ranks.float().cuda())
        else:
            s = self.suit_conv(hot_suits.float())
            r = self.rank_conv(hot_ranks.float())
        x = torch.cat((r,s),dim=-1)
        # x (b, 64, 16)
        for i,hidden_layer in enumerate(self.hidden_layers):
            x = self.activation_fc(hidden_layer(x))
        return self.categorical_output(x.view(M,-1))

class HandRankClassificationFC(nn.Module):
    def __init__(self,params,activation_fc=F.leaky_relu):
        super().__init__()
        self.params = params
        self.device = params['device']
        self.nA = params['nA']
        self.activation_fc = activation_fc
        self.emb_size = 64
        self.seed = torch.manual_seed(params['seed'])
        self.card_emb = nn.Embedding(53,self.emb_size,padding_idx=0)
        self.hidden_dims = params['hidden_dims']
        self.hand_dims = params['hand_dims']
        self.board_dims = params['board_dims']
        # Input is (b,4,2) -> (b,4,4) and (b,4,13)
        self.hand_layers = nn.ModuleList()
        for i in range(len(self.hand_dims)-1):
            self.hand_layers.append(nn.Linear(self.hand_dims[i],self.hand_dims[i+1]))
        self.board_layers = nn.ModuleList()
        for i in range(len(self.board_dims)-1):
            self.board_layers.append(nn.Linear(self.board_dims[i],self.board_dims[i+1]))
        self.hidden_layers = nn.ModuleList()
        for i in range(len(self.hidden_dims)-1):
            self.hidden_layers.append(nn.Linear(self.hidden_dims[i],self.hidden_dims[i+1]))
        self.categorical_output = nn.Linear(self.hidden_dims[-1],self.nA)

    def forward(self,x):
        """
        Emb and process hand, emb and process hand.
        """
        # Input is (b,5) each card is a 53 digit, 0 is padding.
        B,M = x.size()
        cards = self.card_emb(x.long())
        hero_cards = cards[:,:2,:].view(B,-1)
        board_cards = cards[:,2:,:].view(B,-1)
        for hidden_layer in self.hand_layers:
            hero_cards = self.activation_fc(hidden_layer(hero_cards))
        for hidden_layer in self.board_layers:
            board_cards = self.activation_fc(hidden_layer(board_cards))
        x = torch.cat((hero_cards,board_cards),dim=-1)
        # x (b, 64, 32)
        for i,hidden_layer in enumerate(self.hidden_layers):
            x = self.activation_fc(hidden_layer(x))
        return self.categorical_output(x)

################################################
#            Smalldeck Classification          #
################################################

class SmalldeckClassification(nn.Module):
    def __init__(self,params,hidden_dims=(16,32,32),output_dims=(3600,512,256,127),activation_fc=F.relu):
        super().__init__()
        self.params = params
        self.nA = params['nA']
        self.activation_fc = activation_fc
        self.seed = torch.manual_seed(params['seed'])
        self.device = params['device']
        self.output_dims = output_dims
        self.one_hot_suits = torch.nn.functional.one_hot(torch.arange(0,dt.SUITS.HIGH))
        self.one_hot_ranks = torch.nn.functional.one_hot(torch.arange(0,dt.RANKS.HIGH))

        # Input is (b,4,2) -> (b,4,4) and (b,4,13)
        self.suit_conv = nn.Sequential(
            nn.Conv1d(5, 16, kernel_size=1, stride=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )
        self.rank_conv = nn.Sequential(
            nn.Conv1d(5, 16, kernel_size=5, stride=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )
        self.hidden_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i],hidden_dims[i+1]))
            # self.bn_layers.append(nn.BatchNorm1d(64))
        self.categorical_output = nn.Linear(512,7463)
        self.k_dim = 256
        self.query = nn.Linear(self.k_dim,self.k_dim)
        self.keys = nn.Linear(self.k_dim,self.k_dim)
        self.value = nn.Linear(self.k_dim,self.k_dim)
        self.output_layers = nn.ModuleList()
        for i in range(len(self.output_dims)-1):
            self.output_layers.append(nn.Linear(self.output_dims[i],self.output_dims[i+1]))
        self.small_category_out = nn.Linear(128,self.nA)

    def forward(self,x):
        # Expects shape of (B,18)
        x = x.unsqueeze(0)
        B,M,C = x.size()
        ranks,suits = unspool(x)
        # Shape of B,M,60,5
        hot_ranks = self.one_hot_ranks[ranks].float().to(self.device)
        hot_suits = self.one_hot_suits[suits].float().to(self.device)
        # hot_ranks torch.Size([1, 2, 60, 5, 15])
        # hot_suits torch.Size([1, 2, 60, 5, 5])
        # torch.set_printoptions(threshold=7500)
        raw_activations = []
        activations = []
        for i in range(B):
            raw_combinations = []
            combinations = []
            for j in range(M):
                s = self.suit_conv(hot_suits[i,j,:,:,:])
                r = self.rank_conv(hot_ranks[i,j,:,:,:])
                out = torch.cat((r,s),dim=-1)
                out_flat = out.view(60,-1)
                # 60,16,16
                # Self attention
                # q = self.query(out_flat)
                # k = self.keys(out_flat)
                # v = self.value(out_flat)
                # raw = torch.mm(q,k.transpose(0,1)) / math.sqrt(self.k_dim)
                # weights = F.softmax(raw,dim=-1)
                # raw_out = torch.mm(weights,raw)
                # y_hat = torch.mm(raw_out,v)
                raw_combinations.append(out_flat)
                # out: (b,64,16)
                for hidden_layer in self.hidden_layers:
                    out = self.activation_fc(hidden_layer(out))
                out = self.categorical_output(out.view(60,-1))
                combinations.append(torch.argmax(out,dim=-1))
            activations.append(torch.stack(combinations))
            raw_activations.append(torch.stack(raw_combinations))
        # baseline = hardcode_handstrength(x)
        results = torch.stack(activations)
        best_hand = torch.min(results,dim=-1)[0].unsqueeze(-1)
        # print(best_hand)
        # print(baseline)
        raw_results = torch.stack(raw_activations).view(B,M,-1)
        # (B,M,60,7463)
        for output_layer in self.output_layers:
            raw_results = self.activation_fc(output_layer(raw_results))
        # (B,M,60,512)
        # o = self.hand_out(raw_results.view(B,M,-1))
        final_out = torch.cat((raw_results,best_hand.float()),dim=-1)
        return self.small_category_out(final_out.view(M,-1))

class SmalldeckClassificationFlat(nn.Module):
    def __init__(self,params,hidden_dims=(256,256,128),hand_dims=(128,512,128),board_dims=(192,512,128),output_dims=(15360,512,256,127),activation_fc=F.leaky_relu):
        super().__init__()
        self.params = params
        self.nA = params['nA']
        self.activation_fc = activation_fc
        self.seed = torch.manual_seed(params['seed'])
        self.device = params['device']
        self.output_dims = output_dims
        self.emb_size = 64
        self.card_emb = nn.Embedding(53,self.emb_size,padding_idx=0)
        # Input is (b,4,2) -> (b,4,4) and (b,4,13)
        self.hand_layers = nn.ModuleList()
        for i in range(len(hand_dims)-1):
            self.hand_layers.append(nn.Linear(hand_dims[i],hand_dims[i+1]))
        self.board_layers = nn.ModuleList()
        for i in range(len(board_dims)-1):
            self.board_layers.append(nn.Linear(board_dims[i],board_dims[i+1]))
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i],hidden_dims[i+1]))
        self.categorical_output = nn.Linear(128,7463)
        self.output_layers = nn.ModuleList()
        for i in range(len(self.output_dims)-1):
            self.output_layers.append(nn.Linear(self.output_dims[i],self.output_dims[i+1]))
        self.small_category_out = nn.Linear(output_dims[-1],self.nA)

    def forward(self,x):
        # Expects shape of (B,18)
        x = x.unsqueeze(0)
        B,M,C = x.size()
        ranks,suits = unspool(x)
        cards = ((ranks-1)*suits)
        # cards = ((x[:,:,0]+1) * x[:,:,1]).long()
        emb_cards = self.card_emb(cards)
        # Shape of B,M,60,5,64
        raw_activations = []
        activations = []
        for i in range(B):
            raw_combinations = []
            combinations = []
            for j in range(M):
                hero_cards = emb_cards[i,j,:,:2,:].view(60,-1)
                board_cards = emb_cards[i,j,:,2:,:].view(60,-1)
                for hidden_layer in self.hand_layers:
                    hero_cards = self.activation_fc(hidden_layer(hero_cards))
                for hidden_layer in self.board_layers:
                    board_cards = self.activation_fc(hidden_layer(board_cards))
                out = torch.cat((hero_cards,board_cards),dim=-1)
                raw_combinations.append(out)
                # x (b, 64, 32)
                for hidden_layer in self.hidden_layers:
                    out = self.activation_fc(hidden_layer(out))
                combinations.append(torch.argmax(self.categorical_output(out),dim=-1))
            activations.append(torch.stack(combinations))
            raw_activations.append(torch.stack(raw_combinations))
        # baseline = hardcode_handstrength(x)
        results = torch.stack(activations)
        best_hand = torch.min(results,dim=-1)[0].unsqueeze(-1)
        # print(best_hand)
        # print(baseline)
        raw_results = torch.stack(raw_activations).view(B,M,-1)
        # (B,M,60,7463)
        for output_layer in self.output_layers:
            raw_results = self.activation_fc(output_layer(raw_results))
        # (B,M,60,512)
        # final_out = torch.cat((raw_results,best_hand.float()),dim=-1)
        return self.small_category_out(raw_results.view(M,-1))

################################################
#            Partial hand regression           #
################################################

class PartialHandRegression(nn.Module):
    def __init__(self,params,hidden_dims=(16,32,32),activation_fc=F.relu):
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
        return self.categorical_output(x)