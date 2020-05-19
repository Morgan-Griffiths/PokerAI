import torch
import torch.nn as nn
import torch.nn.functional as F
import hand_recognition.datatypes as dt

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