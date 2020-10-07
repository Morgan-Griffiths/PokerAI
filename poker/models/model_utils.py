import torch, os
import numpy as np
from prettytable import PrettyTable
from itertools import combinations

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

def expand_conv2d(network,path):
    layer_weights = torch.load(path)
    for name, param in network.process_input.hand_board.rank_conv.named_parameters():
        print(name,param.shape)
        if len(param.shape) > 1:
            print('loading rank')
            if name == 'weight':
                expanded_param = param.repeat(60,1,1,1).permute(1,0,2,3)
                param.data.copy_(layer_weights.rank_conv.weight.data)
                param.requires_grad = False
            elif name == 'bias':
                param.data.copy_(layer_weights.rank_conv.bias.data)
                param.requires_grad = False
    for name, param in network.process_input.hand_board.suit_conv.named_parameters():
        print(name,param.shape)
        if len(param.shape) > 1:
            print('loading suit')
            if name == 'weight':
                expanded_param = param.repeat(60,1,1,1).permute(1,0,2,3)
                param.data.copy_(layer_weights.rank_conv.weight.data)
                param.requires_grad = False
            elif name == 'bias':
                param.data.copy_(layer_weights.rank_conv.bias.data)
                param.requires_grad = False

def update_weights(network,path):
    layer_weights = torch.load(path)
    for name, param in network.process_input.hand_board.named_parameters():
        if name in layer_weights:
            param.data.copy_(layer_weights[name].data)
            param.requires_grad = False
    return network

def soft_update(local,target,tau=1e-1):
    for local_param,target_param in zip(local.parameters(),target.parameters()):
        target_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)

UNSPOOL_INDEX = np.array([h + b for h in combinations(range(0,4), 2) for b in combinations(range(4,9), 3)])

def unspool(X):
    # Size of (B,M,18)
    ranks = X[:,:,::2]
    suits = X[:,:,1::2]
    sequence_ranks = ranks[:,:,UNSPOOL_INDEX]
    sequence_suits = suits[:,:,UNSPOOL_INDEX]
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
    ranks = torch.arange(14,5,-1)
    suits = torch.arange(1,4).repeat(3)
    combined = torch.zeros(18)
    for i in range(0,9):
        combined[i*2] = ranks[i]
        combined[(i*2)+1] = suits[i]
    combined = combined.repeat(1,2,1)
    unspooled_ranks,unspooled_suits = unspool(combined[0,:,:])