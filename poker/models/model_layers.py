import torch
import torch.nn as nn
import torch.nn.functional as F
from poker_env.datatypes import Globals,SUITS,RANKS,Action,Street,NetworkActions
import numpy as np
from models.model_utils import strip_padding,unspool,flat_unspool,hardcode_handstrength
import time
from functools import lru_cache


class HandBoardClassification(nn.Module):
    def __init__(self,params,hidden_dims=(256,256,128),hand_dims=(128,512,128),board_dims=(192,512,128),output_dims=(15360,512,256,127),activation_fc=F.leaky_relu):
        super().__init__()
        self.params = params
        self.nA = params['nA']
        self.activation_fc = activation_fc
        self.seed = torch.manual_seed(params['seed'])
        self.device = params['device']
        self.output_dims = output_dims
        self.emb_size = 64
        self.seed = torch.manual_seed(params['seed'])
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
        self.categorical_output = nn.Linear(128,self.nA)
        self.output_layers = nn.ModuleList()
        for i in range(len(self.output_dims)-1):
            self.output_layers.append(nn.Linear(self.output_dims[i],self.output_dims[i+1]))
        self.small_category_out = nn.Linear(output_dims[-1],self.nA)

    def forward(self,x):
        B = 1
        M = x.size(1)
        cards = flat_unspool(x)
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


class IdentityBlock(nn.Module):
    def __init__(self,hidden_dims,activation):
        """hidden_dims must contain 3 values"""
        super().__init__()
        assert len(hidden_dims) == 3
        self.activation = activation
        self.fc1 = nn.Linear(hidden_dims[0],hidden_dims[1])
        self.fc2 = nn.Linear(hidden_dims[1],hidden_dims[2])

    def forward(self,x):
        out = self.activation(self.fc1(x)) + x
        out2 = self.activation(self.fc2(out))
        return out2

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

    @lru_cache(maxsize=30)
    def batch_unwrap_action(self,actions:torch.Tensor,previous_actions:torch.Tensor):
        """
        Unwraps flat action into action_category and betsize_category
        Action is from network outputs - 0-5
        previous_action is from env. 1-6
        """
        int_actions = torch.zeros_like(actions)
        int_betsizes = torch.zeros_like(actions)
        prev_ge_2 = torch.as_tensor((previous_actions > Action.FOLD)&(previous_actions < Action.UNOPENED))
        prev_le_3 = torch.as_tensor((previous_actions < Action.CALL)|(previous_actions == Action.UNOPENED))
        actions_le_2 = actions < 2
        actions_ge_2 = actions > 2
        actions_le_3 = actions < 3
        int_actions[actions_le_2] = actions[actions_le_2]
        int_betsizes[actions_le_2] = 0
        int_betsizes[actions_ge_2] = actions[actions_ge_2] - 3
        int_actions[(actions_ge_2)&(prev_ge_2)] = 4
        int_actions[(actions_ge_2)&(prev_le_3)] = 3
        int_actions[actions_le_3] = actions[actions_le_3]
        return int_actions,int_betsizes

    @lru_cache(maxsize=30)
    def unwrap_action(self,action:torch.Tensor,previous_action:torch.Tensor):
        """
        Unwraps flat action into action_category and betsize_category
        Action is from network outputs - 0-5
        previous_action is from env. 1-6
        """
        actions = torch.zeros(self.nA)
        betsizes = torch.zeros(self.nB)
        # actions[action[action < 3]] = 1
        if action < NetworkActions.BET:
            actions[action] = 1
        elif previous_action == Action.UNOPENED or previous_action == Action.CHECK: # Unopened
            actions[3] = 1
            bet_category = action - 3
            betsizes[bet_category] = 1
        else: # facing bet or raise or call (preflop)
            actions[4] = 1
            bet_category = action - 3
            betsizes[bet_category] = 1
        int_actions = torch.argmax(actions, dim=0).unsqueeze(-1)
        int_betsizes = torch.argmax(betsizes, dim=0).unsqueeze(-1)
        return int_actions,int_betsizes

    # def unwrap_action(self,action:torch.Tensor,previous_action:torch.Tensor):
    #     """Unwraps flat action into action_category and betsize_category"""
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

################################################
#              Processing Layers               #
################################################

class ProcessHandBoardConv(nn.Module):
    def __init__(self,params,hand_length,hidden_dims=(16,32,32),output_dims=(15360,512,256,127),activation_fc=F.relu):
        super().__init__()
        self.output_dims = output_dims
        self.activation_fc = activation_fc
        self.hidden_dims = hidden_dims
        self.hand_length = hand_length
        self.one_hot_suits = torch.nn.functional.one_hot(torch.arange(0,SUITS.HIGH))
        self.one_hot_ranks = torch.nn.functional.one_hot(torch.arange(0,RANKS.HIGH))
        self.maxlen = params['maxlen']
        self.device = params['device']
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
        for i in range(len(self.hidden_dims)-1):
            self.hidden_layers.append(nn.Linear(self.hidden_dims[i],self.hidden_dims[i+1]))
        self.categorical_output = nn.Linear(512,7463)
        self.output_layers = nn.ModuleList()
        for i in range(len(self.output_dims)-1):
            self.output_layers.append(nn.Linear(self.output_dims[i],self.output_dims[i+1]))
        # self.hand_out = nn.Linear(128,256) #params['lstm_in'] // 3)

    def set_device(self,device):
        self.device = device

    def forward(self,x):
        """
        x: concatenated hand and board. alternating rank and suit.
        shape: B,M,18
        """
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
                raw_combinations.append(out)
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
        return torch.cat((raw_results,best_hand.float()),dim=-1)

class ProcessHandBoard(nn.Module):
    def __init__(self,params,hand_length,hidden_dims=(256,256),hand_dims=(32,128),board_dims=(48,128),output_dims=(15360,256),activation_fc=F.leaky_relu):
        super().__init__()
        self.params = params
        self.activation_fc = activation_fc
        self.maxlen = params['maxlen']
        self.device = params['device']
        self.hidden_dims = hidden_dims
        self.hand_length = hand_length
        self.output_dims = output_dims
        self.emb_size = 16
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
        self.categorical_output = nn.Linear(256,7463)
        self.output_layers = nn.ModuleList()
        for i in range(len(self.output_dims)-1):
            self.output_layers.append(nn.Linear(self.output_dims[i],self.output_dims[i+1]))

    def forward(self,x):
        # Expects shape of (B,M,18)
        B,M,C = x.size()
        ranks,suits = unspool(x)
        # Flatten cards
        ranks[ranks>0] -= 1
        suits[suits>0] = (suits[suits>0] -1) * 13
        cards = ranks + suits
        emb_cards = self.card_emb(cards)
        # Shape of B,M,60,5,64
        raw_activations = []
        # activations = []
        for i in range(B):
            raw_combinations = []
            # combinations = []
            for j in range(M):
                hero_cards = emb_cards[i,j,:,:2,:].view(60,-1)
                board_cards = emb_cards[i,j,:,2:,:].view(60,-1)
                for hidden_layer in self.hand_layers:
                    hero_cards = self.activation_fc(hidden_layer(hero_cards))
                for hidden_layer in self.board_layers:
                    board_cards = self.activation_fc(hidden_layer(board_cards))
                out = torch.cat((hero_cards,board_cards),dim=-1)
                # x (b, 64, 32)
                for hidden_layer in self.hidden_layers:
                    out = self.activation_fc(hidden_layer(out))
                raw_combinations.append(out)
                # combinations.append(torch.argmax(self.categorical_output(out),dim=-1))
            # activations.append(torch.stack(combinations))
            raw_activations.append(torch.stack(raw_combinations))
        # results = torch.stack(activations)
        # baseline = hardcode_handstrength(x)
        # best_hand = torch.flip(torch.min(results,dim=-1)[0].unsqueeze(-1),dims=(0,1))
        # best_hand = torch.min(results,dim=-1)[0].unsqueeze(-1)
        raw_results = torch.stack(raw_activations).view(B,M,-1)
        # (B,M,60,7463)
        for output_layer in self.output_layers:
            raw_results = self.activation_fc(output_layer(raw_results))
        return raw_results
        # (B,M,60,512)
        # return torch.cat((raw_results,best_hand.view(B,M,-1).float()),dim=-1)

class ProcessOrdinal(nn.Module):
    def __init__(self,critic,params,activation_fc=F.relu):
        super().__init__()
        self.activation_fc = activation_fc
        self.critic = critic
        self.device = params['device']
        self.street_emb = nn.Embedding(embedding_dim=params['embedding_size']//4, num_embeddings=Street.RIVER+1,padding_idx=0)
        self.action_emb = nn.Embedding(embedding_dim=params['embedding_size']//4, num_embeddings=Action.UNOPENED+1,padding_idx=0)
        self.position_emb = nn.Embedding(embedding_dim=params['embedding_size']//4, num_embeddings=4,padding_idx=0)
        self.actor_indicies = {
            'hero_pos':0,
            'street':2,
            'last_action_pos':4,
            'prev_action':5
        }
        self.critic_indicies = {
            'hero_pos':0,
            'street':1,
            'last_action_pos':5,
            'prev_action':6
        }
        if critic:
            self.indicies = self.critic_indicies
        else:
            self.indicies = self.actor_indicies

    def forward(self,x):
        hero_position = self.street_emb(x[:,:,self.indicies['hero_pos']].long())
        street = self.street_emb(x[:,:,self.indicies['street']].long())
        previous_action = self.action_emb(x[:,:,self.indicies['prev_action']].long())
        last_action_position = self.position_emb(x[:,:,self.indicies['last_action_pos']].long())
        return torch.cat((street,hero_position,previous_action,last_action_position),dim=-1)

class ProcessContinuous(nn.Module):
    def __init__(self,critic,params,activation_fc=F.relu):
        super().__init__()
        self.activation_fc = activation_fc
        self.stack_fc = nn.Linear(1,params['embedding_size']//4)
        self.call_fc = nn.Linear(1,params['embedding_size']//4)
        self.odds_fc = nn.Linear(1,params['embedding_size']//4)
        self.pot_fc = nn.Linear(1,params['embedding_size']//4)
        self.critic_indicies = {
            'hero_stack':0,
            'pot':3,
            'amnt_to_call':4,
            'pot_odds':5
        }
        self.actor_indicies = {
            'hero_stack':0,
            'pot':4,
            'amnt_to_call':5,
            'pot_odds':6
        }
        if critic:
            self.indicies = self.critic_indicies
        else:
            self.indicies = self.actor_indicies

    def forward(self,x):
        B,M,C = x.size()
        hero_stack = x[:,:,self.indicies['hero_stack']]
        amnt_to_call = x[:,:,self.indicies['amnt_to_call']]
        pot_odds = x[:,:,self.indicies['pot_odds']]
        pot = x[:,:,self.indicies['pot']]
        stack = []
        calls = []
        odds = []
        pot = []
        for i in range(B):
            for j in range(M):
                stack.append(self.activation_fc(self.stack_fc(hero_stack[i,j].unsqueeze(-1))))
                calls.append(self.activation_fc(self.call_fc(amnt_to_call[i,j].unsqueeze(-1))))
                odds.append(self.activation_fc(self.odds_fc(pot_odds[i,j].unsqueeze(-1))))
                pot.append(self.activation_fc(self.pot_fc(pot_odds[i,j].unsqueeze(-1))))
        emb_pot = torch.stack(pot).view(B,M,-1)
        emb_call = torch.stack(calls).view(B,M,-1)
        emb_stack = torch.stack(stack).view(B,M,-1)
        emb_odds = torch.stack(odds).view(B,M,-1)
        if emb_pot.dim() == 2:
            emb_pot = emb_pot.unsqueeze(0)
            emb_call = emb_call.unsqueeze(0)
            emb_stack = emb_stack.unsqueeze(0)
            emb_odds = emb_odds.unsqueeze(0)
        return torch.stack((emb_pot,emb_call,emb_stack,emb_odds),dim=-1).view(B,M,-1)

class PreProcessLayer(nn.Module):
    def __init__(self,params,critic=False):
        super().__init__()
        self.critic = critic
        self.maxlen = params['maxlen']
        self.state_mapping = params['state_mapping']
        self.obs_mapping = params['obs_mapping']
        self.device = params['device']
        hand_length = Globals.HAND_LENGTH_DICT[params['game']]
        self.hand_board = ProcessHandBoard(params,hand_length)
        self.continuous = ProcessContinuous(critic,params)
        self.ordinal = ProcessOrdinal(critic,params)

    def set_device(self,device):
        self.device = device
        self.hand_board.set_device(device)
        
    def forward(self,x):
        B,M,C = x.size()
        if self.critic:
            h1 = self.hand_board(x[:,:,self.obs_mapping['hand_board']].float())
            h2 = self.hand_board(x[:,:,self.obs_mapping['villain_board']].float())
            h = h1 - h2
            o = self.ordinal(x[:,:,self.obs_mapping['ordinal']].to(self.device))
            c = self.continuous(x[:,:,self.obs_mapping['continuous']].to(self.device))
        else:
            h = self.hand_board(x[:,:,self.state_mapping['hand_board']].float())
            o = self.ordinal(x[:,:,self.state_mapping['ordinal']].to(self.device))
            c = self.continuous(x[:,:,self.state_mapping['continuous']].to(self.device))
        combined = torch.cat((h,o,c),dim=-1)
        return combined

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

    def __init__(self,device='cpu', sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0).float().to(device)

    def set_device(self,device):
        self.noise = self.noise.to(device)

    def forward(self, x):
        if self.training and self.sigma != 0:
            #x = x.cpu()
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x#.cuda()

class EncoderAttention(nn.Module):
    def __init__(self,in_size,lstm_out):
        super().__init__()
        self.context_nn = nn.Linear(lstm_out,in_size)
        
    def forward(self,x,hidden_states):
        context = self.context_nn(hidden_states)
        scores = F.softmax(context,dim=-1)
        return scores * x

class VectorAttention(nn.Module):
    def __init__(self,in_size):
        super().__init__()
        self.context_nn = nn.Linear(in_size,in_size)
        
    def forward(self,x):
        context = self.context_nn(x)
        scores = F.softmax(context,dim=-1)
        return scores * x

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