import torch
import numpy as np


def combined_masks(action_mask,betsize_mask):
    """Combines action and betsize masks into flat mask for 1d network outputs"""
    if action_mask.dim() > 2:
        return torch.cat([action_mask[:,:,:-2],betsize_mask],dim=-1)
    elif action_mask.dim() > 1:
        return torch.cat([action_mask[:,:-2],betsize_mask],dim=-1)
    else:
        return torch.cat([action_mask[:-2],betsize_mask])

def strip_padding(x,maxlen):
    assert(x.ndim == 3)
    mask = np.where(x.sum(-1).numpy() == 0)
    padding = np.unique(mask[0],return_counts=True)[1]
    n_states = torch.tensor(maxlen - padding)
    return x[:,:n_states,:]

def norm_frequencies(action_soft,mask):
    # with torch.no_grad():
    action_masked = action_soft * mask
    action_probs =  action_masked / action_masked.sum(-1).unsqueeze(-1)
    return action_probs

def hard_update(source,target):
    for target_param,param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)