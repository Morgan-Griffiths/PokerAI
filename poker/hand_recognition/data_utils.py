import torch
import numpy as np
import os

import hand_recognition.datatypes as dt
from hand_recognition.build_data import CardDataset
from utils import torch_where


def return_handtype_dict(X:torch.tensor,y:torch.tensor):
    type_dict = {}
    for key in dt.Globals.HAND_TYPE_DICT.keys():
        if key == 0:
            mask = torch.zeros_like(y)
            mask[(y == 0).nonzero().unsqueeze(0)] = 1
            type_dict[key] = mask
        else:
            type_dict[key] = torch_where(y==key,y)
        assert(torch.max(type_dict[key]).item() == 1)
    return type_dict
    
def load_data(dir_path='data/predict_winner'):
    data = {}
    for f in os.listdir(dir_path):
        if f != '.DS_store':
            name = os.path.splitext(f)[0]
            data[name] = torch.Tensor(np.load(os.path.join(dir_path,f)))
    return data

def save_data(trainX,trainY,valX,valY,params):
    if not os.path.isdir(params['save_path']):
        os.makedirs(params['save_path'])
    np.save(f"{params['save_path']}/trainX",trainX)
    np.save(f"{params['save_path']}/trainY",trainY)
    np.save(f"{params['save_path']}/valX",valX)
    np.save(f"{params['save_path']}/valY",valY)

def unpack_nparrays(shape,batch,data):
    X = np.zeros(shape)
    Y = np.zeros(shape[0])
    i = 0
    j = 0
    for k,v in data.items():
        Y[i*batch:(i+1)*batch] = dt.Globals.HAND_TYPE_FILE_DICT[k]
        for hand in v:
            X[j] = np.stack(hand)
            j += 1
        i += 1
    print('Numpy data uniques and counts ',np.lib.arraysetops.unique(Y,return_counts=True))
    return torch.tensor(X),torch.tensor(Y).long()
