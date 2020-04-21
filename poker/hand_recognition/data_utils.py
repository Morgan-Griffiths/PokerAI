import torch
import numpy as np
import os

import hand_recognition.datatypes as dt
from hand_recognition.build import CardDataset
from utils import torch_where

def return_handtype_dict(X:torch.tensor,y:torch.tensor,target_dict=dt.Globals.HAND_TYPE_DICT):
    type_dict = {}
    print(np.unique(y,return_counts=True))
    for key in target_dict.keys():
        type_dict[key] = torch.tensor(np.where(y.numpy() == key)[0])
    return type_dict


def load_data(dir_path):
    """
    loads train,test folder numpy data from parent dir
    """
    print(dir_path)
    data = {}
    for folder in os.listdir(dir_path):
        if folder != '.DS_store':
            for f in os.listdir(os.path.join(dir_path,folder)):
                name = os.path.splitext(f)[0]
                data[name] = torch.Tensor(np.load(os.path.join(dir_path,folder,f)))
    return data

def load_handtypes(dir_path):
    """
    loads train,test folder numpy data from parent dir
    """
    data = {}
    for folder in os.listdir(dir_path):
        if folder != '.DS_store':
            data[folder] = {}
            print(folder)
            for f in os.listdir(os.path.join(dir_path,folder)):
                name = os.path.splitext(f)[0]
                data[folder][name] = torch.Tensor(np.load(os.path.join(dir_path,folder,f)))
    return data

def save_data(trainX,trainY,valX,valY,parent_dir):
    """
    saves train,test folder numpy data from parent dir
    """
    if not os.path.isdir(parent_dir):
        os.makedirs(parent_dir)
    np.save(f"{os.path.join(parent_dir,'train')}/trainX",trainX)
    np.save(f"{os.path.join(parent_dir,'train')}/trainY",trainY)
    np.save(f"{os.path.join(parent_dir,'test')}/valX",valX)
    np.save(f"{os.path.join(parent_dir,'test')}/valY",valY)

def unpack_nparrays(shape,batch,data):
    """
    Takes numpy array data in the form of X inputs and file names, 
    and builds the Y targets given the filenames
    """
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
    return torch.tensor(X).float(),torch.tensor(Y).long()

def return_handtype_data_shapes(dataset:dict):
    """
    Assumes each handtype is weighted equally
    """
    for key in dataset.keys():
        trm,trh,trw = dataset[key].size()
        break
    train_shape = (trm*9,trh,trw)
    train_batch = trm
    return train_shape,train_batch