import os
import torch

def save_weights(self,network,path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.mkdir(directory)
    torch.save(network.state_dict(), path)