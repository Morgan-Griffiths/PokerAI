import torch, os
import numpy as np

def update_weights(networks,path):
    layer_weights = torch.load(path)
    for network in networks:
        for name, param in network.process_input.hand_board.named_parameters():
            param.data.copy_(layer_weights[name].data)
            param.requires_grad = False
    return networks

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