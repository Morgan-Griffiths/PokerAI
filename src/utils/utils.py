import numpy as np
import pickle
from torch import where,zeros_like

def return_uniques(values):
    uniques, count = np.unique(values, return_counts=True)
    return uniques, count

def torch_where(condition,vec):
    mask = where(condition,vec,zeros_like(vec))
    mask[mask>0]= 1
    return mask

def savepickle(data,path):
    #store data
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol = pickle.HIGHEST_PROTOCOL)   
        
def openpickle(path):
    #to read file
    with open(path, 'rb') as handle:
        b = pickle.load(handle)
    return b