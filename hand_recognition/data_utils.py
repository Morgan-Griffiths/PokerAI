import torch
import numpy as np
import os

def unspool(X):
    # Size of (M,9,2)
    M = X.size(0)
    hand = X[:,:4,:].permute(1,0,2)
    hand_combos = combinations(hand,2)
    board = X[:,4:,:].permute(1,0,2)
    board_combos = list(combinations(board,3))
    combined = torch.zeros(M,60,5,2)
    i = 0
    for hcombo in hand_combos:
        for bcombo in board_combos:
            stacked_board = torch.stack(bcombo)
            stacked_hand = torch.stack(hcombo)
            combined[:,i,:,:] = torch.cat((stacked_hand[torch.argsort(stacked_hand[:,0,0]),:,:],stacked_board[torch.argsort(stacked_board[:,0,0]),:,:]),dim=0).permute(1,0,2)
            i += 1
    return combined

def return_ylabel_dict(X:torch.tensor,y:torch.tensor,target_set:set):
    type_dict = {}
    # print(np.unique(y,return_counts=True))
    for item in target_set:
        type_dict[item] = torch.tensor(np.where(y == item)[0])
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

def save_trainset(trainX,trainY,parent_dir,y_dtype='uint8'):
    """
    saves train,test folder numpy data from parent dir
    """
    train_dir = os.path.join(parent_dir,'train')
    if not os.path.isdir(train_dir):
        os.makedirs(train_dir)
    np.save(f"{os.path.join(parent_dir,'train')}/trainX",np.array(trainX).astype('uint8'))
    np.save(f"{os.path.join(parent_dir,'train')}/trainY",np.array(trainY).astype(y_dtype))

def save_all(trainX,trainY,valX,valY,parent_dir,y_dtype='uint8'):
    """
    saves train,test folder numpy data from parent dir
    """
    train_dir = os.path.join(parent_dir,'train')
    test_dir = os.path.join(parent_dir,'val')
    if not os.path.isdir(train_dir):
        os.makedirs(train_dir)
    if not os.path.isdir(test_dir):
        os.makedirs(test_dir)
    np.save(f"{os.path.join(parent_dir,'train')}/trainX",np.array(trainX).astype('uint8'))
    np.save(f"{os.path.join(parent_dir,'train')}/trainY",np.array(trainY).astype(y_dtype))
    np.save(f"{os.path.join(parent_dir,'val')}/valX",np.array(valX).astype('uint8'))
    np.save(f"{os.path.join(parent_dir,'val')}/valY",np.array(valY).astype(y_dtype))

def save_data(data,path):
    directory = os.path.dirname(path)
    if not os.path.isdir(directory):
        os.makedirs(directory)
    print('save_data',path)
    np.save(path,data.astype('uint8'))

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