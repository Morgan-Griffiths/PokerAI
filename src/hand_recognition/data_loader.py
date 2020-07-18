import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import train_test_split,StratifiedKFold

from utils.utils import return_uniques

class datasetLoader(Dataset):
    """Boolean Logic dataset."""

    def __init__(self, X,y):
        if isinstance(X,(np.generic,np.ndarray)):
            self.X = torch.from_numpy(X)
            self.y = torch.from_numpy(y)
        else:
            self.X = X
            self.y = y
        print(self.X.shape,self.y.shape)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample = {'item': self.X[idx], 'label': self.y[idx]}

        return sample

def return_trainloader(X,y):
    data = datasetLoader(X,y)
    params = {
        'batch_size':2048,
        'shuffle': True,
        'num_workers':2
    }
    trainloader = DataLoader(data, **params)
    return trainloader

def return_dataloaders(X,y,multigpu=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1)
    print(f'Dataset binary label balance: train: {return_uniques(y_train)[1]}, val: {return_uniques(y_val)[1]}, test: {return_uniques(y_test)[1]}')

    trainset = datasetLoader(X_train,y_train)
    valset = datasetLoader(X_val,y_val)
    testset = datasetLoader(X_test,y_test)

    params = {
        'batch_size':2048,
        'shuffle': True,
        'num_workers':2
    }
    trainloader = DataLoader(trainset, **params)
    valloader = DataLoader(valset,**params)
    params['shuffle'] = False
    testloader = DataLoader(testset, **params)

    data = {
        'trainset':trainloader,
        'valset':valloader,
        'testset':testloader
    }
    return data