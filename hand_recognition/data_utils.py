import torch
import numpy as np
import os

def return_ylabel_dict(X:torch.tensor,y:torch.tensor,target_set:set):
    type_dict = {}
    print(np.unique(y,return_counts=True))
    for item in target_set:
        type_dict[item] = torch.tensor(np.where(y.numpy() == item)[0])
    return type_dict


def load_data(dir_path):
    """
    loads train,val,test folder numpy data from parent dir
    """
    print(dir_path)
    data = {}
    for folder in os.listdir(dir_path):
        if folder != '.DS_store':
            for f in os.listdir(os.path.join(dir_path,folder)):
                name = os.path.splitext(f)[0]
                data[name] = np.load(os.path.join(dir_path,folder,f),mmap_mode='r')
    return data

def load_handtypes(dir_path):
    """
    loads train,,val,test folder numpy data from parent dir
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

def save_all(trainX,trainY,valX,valY,parent_dir):
    """
    saves train,test folder numpy data from parent dir
    """
    train_dir = os.path.join(parent_dir,'train')
    test_dir = os.path.join(parent_dir,'val')
    if not os.path.isdir(train_dir):
        os.makedirs(train_dir)
    if not os.path.isdir(test_dir):
        os.makedirs(test_dir)
    np.save(f"{os.path.join(parent_dir,'train')}/trainX",trainX)
    np.save(f"{os.path.join(parent_dir,'train')}/trainY",trainY)
    np.save(f"{os.path.join(parent_dir,'val')}/valX",valX)
    np.save(f"{os.path.join(parent_dir,'val')}/valY",valY)

def save_data(data,path):
    directory = os.path.dirname(path)
    if not os.path.isdir(directory):
        os.makedirs(directory)
    np.save(path,data)
    

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