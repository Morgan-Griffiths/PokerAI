import torch, os
import numpy as np
from prettytable import PrettyTable
from itertools import combinations
from utils.cardlib import hand_rank,encode
from collections import OrderedDict

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

def strip_module(path):
    state_dict = torch.load(path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == 'module.':
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def load_weights(net,path,id=0,ddp=False):
    if torch.cuda.is_available():
        # check if module is in the dict name
        if ddp:
            map_location = {'cuda:%d' % 0: 'cuda:%d' % id}
            net.load_state_dict(torch.load(path,map_location=map_location))
        else:
            net.load_state_dict(strip_module(path))
    else: 
        net.load_state_dict(torch.load(path,map_location=torch.device('cpu')))

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

def swap_suits(cards):
    """
    Swap suits to remove most symmetries.
    """
    cards_need_swap = cards
    new_suit = 5
    while cards_need_swap.shape[0] > 0:
        suit = cards_need_swap[0,1]
        cards[cards[:,1] == suit, 1] = new_suit
        new_suit += 1
        cards_need_swap = cards[cards[:,1] < 5]
    cards[:,1] = cards[:,1] - 4
    return cards

def compare_weights(net):
    path = '/Users/morgan/Code/PokerAI/poker/checkpoints/frozen_layers/hand_board_weights'
    if torch.cuda.is_available():
        layer_weights = torch.load(path)
    else:
        layer_weights = torch.load(path,map_location=torch.device('cpu'))
    print(net)
    for name, param in net.named_parameters():
        if name in layer_weights:
            # print(param.data == layer_weights[name].data)
            print(f'Layer {name},Equal {np.array_equal(param.data.numpy(),layer_weights[name].data.numpy())}')

def expand_conv2d(network,path):
    if torch.cuda.is_available():
        layer_weights = torch.load(path)
    else:
        layer_weights = torch.load(path,map_location=torch.device('cpu'))
    for name, param in network.process_input.hand_board.rank_conv.named_parameters():
        print(name,param.shape)
        if name in ['0.weight','0.bias']:
            print(f'loading {name}')
            if name == '0.weight':
                expanded_param = layer_weights['rank_conv.0.weight'].data.repeat(60,1,1,1).permute(1,0,2,3)
                # Size 64,60,5,5
                param.data.copy_(expanded_param)
                param.requires_grad = False
            elif name == '0.bias':
                param.data.copy_(layer_weights['rank_conv.0.bias'].data)
                param.requires_grad = False
    for name, param in network.process_input.hand_board.suit_conv.named_parameters():
        print(name,param.shape)
        if name in ['0.weight','0.bias']:
            print(f'loading {name}')
            if name == '0.weight':
                expanded_param = layer_weights['suit_conv.0.weight'].data.repeat(60,1,1,1).permute(1,0,2,3)
                # Size 64,60,5,1
                param.data.copy_(expanded_param)
                param.requires_grad = False
            elif name == '0.bias':
                param.data.copy_(layer_weights['suit_conv.0.bias'].data)
                param.requires_grad = False

def save_eval(module):
    if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.eval()

def copy_weights(network,path):
    # path = '/Users/morgan/Code/PokerAI/poker/checkpoints/frozen_layers/hand_board_weights'
    if torch.cuda.is_available():
        layer_weights = torch.load(path)
    else:
        layer_weights = torch.load(path,map_location=torch.device('cpu'))
    for name, param in network.process_input.hand_board.named_parameters():
        if name in layer_weights:
            print('copying weights',name)
            param.data.copy_(layer_weights[name].data)
            param.requires_grad = False

def soft_update(local,target,device,tau=5e-2):
    for local_param,target_param in zip(local.parameters(),target.parameters()):
        target_param.data.copy_(tau*local_param.data.to(device) + (1-tau)*target_param.data)

UNSPOOL_INDEX = np.array([h + b for h in combinations(range(0,4), 2) for b in combinations(range(4,9), 3)])

def hardcode_handstrength(x):
    """input shape: (b,m,18)"""
    B,M,C = x.size()
    hands = x[:,:,:8]
    boards = x[:,:,8:]
    ranks = []
    for j in range(B):
        for i in range(M):
            hand = [[int(hands[j,i,c]),int(hands[j,i,c+1])] for c in range(0,len(hands[j,i,:]),2)]
            board = [[int(boards[j,i,c]),int(boards[j,i,c+1])] for c in range(0,len(boards[j,i,:]),2)]
            hand_en = [encode(c) for c in hand]
            board_en = [encode(c) for c in board]
            ranks.append(torch.tensor(hand_rank(hand_en,board_en),dtype=torch.float))
    return torch.stack(ranks).view(B,M,1)

def swap_suit_vector(cards):
    """Takes flat suit vector"""
    cards_need_swap = cards
    new_suit = 5
    while cards_need_swap.shape[0] > 0:
        suit = cards_need_swap[0]
        cards[cards[:] == suit] = new_suit
        new_suit += 1
        cards_need_swap = cards[cards[:] < 5]
    cards -= 4
    return cards

def swap_batch_suits(suit_vector):
    batch,seq,c,p = suit_vector.size()
    for b in range(batch):
        for s in range(seq):
            for i in range(60):
                suit_vector[b,s,i,:] = swap_suit_vector(suit_vector[b,s,i,:])
    return suit_vector

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

def return_value_mask(actions):
    """supports both batch and single int actions"""
    if hasattr(actions,'shape'):
        M = actions.shape[0]
    else:
        M = 1
    value_mask = torch.zeros(M,5)
    value_mask[torch.arange(M),actions] = 1
    value_mask = value_mask.bool()
    return value_mask.squeeze(0)

def scale_rewards(reward,min_reward,max_reward,factor=1):
    """Scales rewards between -1 and 1, with optional factor to increase valuation differences"""
    span = (max_reward - min_reward) / 2
    sub = (max_reward+min_reward) / 2
    return ((reward-sub) / span) * factor

def strip_padding(x,maxlen):
    assert(x.ndim == 3)
    mask = np.where(x.sum(-1).numpy() == 0)
    padding = np.unique(mask[0],return_counts=True)[1]
    n_states = torch.tensor(maxlen - padding)
    return x[:,:n_states,:]

def padding_index(x,maxlen):
    assert(x.ndim == 3)
    mask = np.where(x.sum(-1).numpy() == 0)
    padding = np.unique(mask[0],return_counts=True)[1]
    n_states = torch.tensor(maxlen - padding)
    return n_states

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
    action_probs =  action_masked / action_masked.sum(-1).unsqueeze(-1)
    return action_probs

def combined_masks(action_mask,betsize_mask):
    """Combines action and betsize masks into flat mask for 1d network outputs"""
    if action_mask.dim() > 2:
        return torch.cat([action_mask[:,:,:-2],betsize_mask],dim=-1)
    elif action_mask.dim() > 1:
        return torch.cat([action_mask[:,:-2],betsize_mask],dim=-1)
    else:
        return torch.cat([action_mask[:-2],betsize_mask])

def mask_(matrices, maskval=0.0, mask_diagonal=True):
    """
    Masks out all values in the given batch of matrices where i <= j holds,
    i < j if mask_diagonal is false
    In place operation
    :param tns:
    :return:
    """

    b, h, w = matrices.size()

    indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    matrices[:, indices[0], indices[1]] = maskval

def d(tensor=None):
    """
    Returns a device string either for the best available device,
    or for the device corresponding to the argument
    :param tensor:
    :return:
    """
    if tensor is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return 'cuda' if tensor.is_cuda else 'cpu'

def here(subpath=None):
    """
    :return: the path in which the package resides (the directory containing the 'former' dir)
    """
    if subpath is None:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

    return os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', subpath))

def contains_nan(tensor):
    return bool((tensor != tensor).sum() > 0)

if __name__ == '__main__':
    ranks = torch.arange(5,14,1)
    suits = torch.arange(1,4).repeat(3)
    combined = torch.zeros(18)
    for i in range(0,9):
        combined[i*2] = ranks[i]
        combined[(i*2)+1] = suits[i]
    combined = combined.repeat(1,2,1)
    unspooled_ranks,unspooled_suits = unspool(combined)
    # print('unspooled_ranks',unspooled_ranks)
    # print('unspooled_suits',unspooled_suits)