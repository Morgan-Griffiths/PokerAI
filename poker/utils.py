import numpy as np
from torch import where,zeros_like

def return_uniques(values):
    uniques, count = np.unique(values, return_counts=True)
    return uniques, count

def torch_where(condition,vec):
    mask = where(condition,vec,zeros_like(vec))
    mask[mask>0]= 1
    return mask