import torch.nn.functional as F
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import MultiStepLR,StepLR
import torch
import numpy as np
import os
import time
import sys
import copy
from torch import optim
from random import shuffle
from collections import deque

from plot import plot_data
from data_loader import return_trainloader
import datatypes as dt
from networks import *
from network_config import NetworkConfig
from data_utils import load_data,return_ylabel_dict,load_handtypes,return_handtype_data_shapes,unspool,generate_category_weights

CUDA_DICT = {0:'cuda:0',1:'cuda:1'}

def setup_world(rank,world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_classification(dataset_params,agent_params,training_params):
    dataset = load_data(dataset_params['data_path'])
    trainloader = return_trainloader(dataset['trainX'],dataset['trainY'],category='classification')
    valloader = return_trainloader(dataset['valX'],dataset['valY'],category='classification')
    category_weights = generate_category_weights()
    data_dict = {
        'trainloader':trainloader,
        'valloader':valloader,
        'category_weights':category_weights
    }
    print('Data shapes',dataset['trainX'].shape,dataset['trainY'].shape,dataset['valX'].shape,dataset['valY'].shape)
    world_size = 2
    mp.spawn(train_network,
        args=(data_dict,agent_params,training_params,),
        nprocs=world_size,
        join=True)

def train_network(id,data_dict,agent_params,training_params):
    setup_world(id,2)
    agent_params['network_params']['device'] = id
    # device = agent_params['network_params']['gpu1']
    net = training_params['network'](agent_params['network_params'])
    if training_params['resume']:
        load_weights(net)
    count_parameters(net)
    # if torch.cuda.device_count() > 1:
    #     dist.init_process_group("gloo", rank=rank, world_size=world_size)
    net = DDP(net)
    net.to(id)
    if 'category_weights' in data_dict:
        criterion = training_params['criterion'](data_dict['category_weights'].to(id))
    else:
        criterion = training_params['criterion']()
    optimizer = optim.Adam(net.parameters(), lr=0.003)
    lr_stepsize = training_params['epochs'] // 5
    lr_stepper = MultiStepLR(optimizer=optimizer,milestones=[lr_stepsize*2,lr_stepsize*3,lr_stepsize*4],gamma=0.1)
    scores = []
    val_scores = []
    score_window = deque(maxlen=100)
    val_window = deque(maxlen=100)
    for epoch in range(training_params['epochs']):
        losses = []
        for i, data in enumerate(data_dict['trainloader'], 1):
            sys.stdout.write('\r')
            # get the inputs; data is a list of [inputs, targets]
            inputs, targets = data.values()
            inputs = inputs.to(id)
            targets = targets.to(id)
            # targets = targets.cuda() if torch.cuda.is_available() else targets
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            sys.stdout.write("[%-60s] %d%%" % ('='*(60*(i+1)//len(data_dict['trainloader'])), (100*(i+1)//len(data_dict['trainloader']))))
            sys.stdout.flush()
            sys.stdout.write(f", training sample {(i+1):.2f}")
            sys.stdout.flush()
        print('outputs',outputs.shape)
        print(f'\nMaximum value {torch.max(torch.softmax(outputs,dim=-1),dim=-1)[0][:100]}, Location {torch.argmax(torch.softmax(outputs,dim=-1),dim=-1)[:100]}')
        print('targets',targets[:100])
        lr_stepper.step()
        score_window.append(loss.item())
        scores.append(np.mean(score_window))
        print(f'Loss: {np.mean(score_window)}')
    cleanup()

def example(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # create default process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # create local model
    model = nn.Linear(10, 10).to(rank)
    # construct DDP model
    ddp_model = DDP(model, device_ids=[rank])
    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    # forward pass
    outputs = ddp_model(torch.randn(20, 10).to(rank))
    labels = torch.randn(20, 10).to(rank)
    # backward pass
    loss_fn(outputs, labels).backward()
    # update parameters
    optimizer.step()
    cleanup()

def main():
    world_size = 2
    mp.spawn(example,
        args=(world_size,),
        nprocs=world_size,
        join=True)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description=
        """
        effective multiprocessing
        """)
    parser.add_argument('-e','--epochs',
                        help='Number of training epochs',
                        default=10,type=int)
    parser.add_argument('--resume',
                        help='resume training from an earlier run',
                        action='store_true')
    parser.add_argument('--function','-f',
                        metavar='example,',
                        dest='function',
                        help='which function to run'
                        )
    parser.set_defaults(resume=False)
    args = parser.parse_args()

    datatype = 'handranksfive'
    learning_category = dt.Globals.DatasetCategories[datatype]
    network = HandRankClassificationFC#NetworkConfig.DataModels[datatype]
    network_name = HandRankClassificationFC.__name__#NetworkConfig.DataModels[datatype].__name__
    network_path = os.path.join('checkpoints',learning_category,network_name)
    print(f'Loading model {network_path}')

    examine_params = {
        'network':network,
        'load_path':network_path
    }
    dataset_params = {
        'datatype':datatype,
        'learning_category':learning_category,
        'data_path':os.path.join('data',dt.Globals.DatasetCategories[datatype],datatype)
    }
    agent_params = {
        'learning_rate':2e-3,
        'network':network,
        'save_dir':'checkpoints',
        'save_path':network_path,
        'load_path':network_path
    }
    network_params = {
        'seed':346,
        'state_space':(13,2),
        'nA':dt.Globals.ACTION_SPACES[datatype],
        'channels':13,
        'kernel':2,
        'batchnorm':True,
        'conv_layers':1,
        'gpu1': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        'gpu2': torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    }
    training_params = {
        'resume':args.resume,
        'epochs':args.epochs,
        'five_card_conversion':False,
        'one_hot':False,
        'criterion':NetworkConfig.LossFunctions[dataset_params['learning_category']],
        'network': network,
        'save_path':network_path,
        'labels':dt.Globals.LABEL_DICT[datatype],
        'gpu1': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        'gpu2': torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
    }
    agent_params['network_params'] = network_params
    agent_params['examine_params'] = examine_params
    if args.function == 'example':
        main()
    else:
        train_classification(dataset_params,agent_params,training_params)
