import torch.nn.functional as F
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import MultiStepLR,StepLR
import torch
import numpy as np
import os
import yaml
import time
import sys
import copy
from pymongo import MongoClient
from torch import optim
from random import shuffle
from collections import deque,OrderedDict

from plot import plot_data
from data_loader import return_trainloader
import datatypes as dt
from networks import *
from network_config import NetworkConfig
from data_utils import load_data,return_ylabel_dict,load_handtypes,return_handtype_data_shapes,unspool,generate_category_weights

"""
Creating a hand dataset for training and evaluating networks.
Full deck
Omaha
"""
def setup_world(rank,world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def strip_module(path):
    state_dict = torch.load(path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == 'module.':
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def add_module(path):
    state_dict = torch.load(path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = 'module.'+k
        new_state_dict[name] = v
    return new_state_dict

def is_path_ddp(path):
    is_ddp = False
    state_dict = torch.load(path)
    for k in state_dict.keys():
        if k[:7] == 'module.':
            is_ddp = True
        break
    return is_ddp

def is_net_ddp(net):
    is_ddp = False
    for name,param in net.named_parameters():
        if name[:7] == 'module.':
            is_ddp = True
        break
    return is_ddp

def load_weights(net,path,rank=0,ddp=False):
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    if torch.cuda.is_available():
        # check if module is in the dict name
        if is_net_ddp(net):
            if not is_path_ddp(path):
                net.load_state_dict(add_module(path))
            else:
                net.load_state_dict(torch.load(path,map_location=map_location))
        else:
            if not is_path_ddp(path):
                net.load_state_dict(torch.load(path,map_location=map_location))
            else:
                net.load_state_dict(strip_module(path))
    else: 
        net.load_state_dict(torch.load(path,map_location=torch.device('cpu')))

def train_network(id,data_dict,agent_params,training_params):
    print(f'Process {id}')
    if torch.cuda.device_count() > 1:
        setup_world(id,2)
    if torch.cuda.device_count() == 0:
        id = 'cpu'
    agent_params['network_params']['device'] = id
    net = training_params['network'](agent_params['network_params'])
    if training_params['resume']:
        print(f"Loading weights from {training_params['load_path']}")
        load_weights(net,training_params['load_path'])
    if training_params['frozen']:
        print('Loading frozen layer')
        conv_path = '../poker/checkpoints/frozen_layers/hand_board_weights_conv'
        fc_path = 'checkpoints/multiclass_categorization/HandRankClassificationFC'
        copy_weights(net,conv_path)
    count_parameters(net)
    if torch.cuda.device_count() > 1:
        net = DDP(net)
    net.to(id)
    if 'category_weights' in data_dict:
        print('using category weights')
        criterion = training_params['criterion'](data_dict['category_weights'].to(id),reduction='sum')
    else:
        criterion = training_params['criterion']()
    optimizer = optim.Adam(net.parameters(), lr=agent_params['learning_rate'])
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
            # zero the parameter gradients
            optimizer.zero_grad()
            # unspool hand into 60,5 combos
            if training_params['five_card_conversion'] == True:
                inputs = unspool(inputs)
            if training_params['one_hot'] == True:
                inputs = torch.nn.functional.one_hot(inputs)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            sys.stdout.write("[%-60s] %d%%" % ('='*(60*(i+1)//len(data_dict['trainloader'])), (100*(i+1)//len(data_dict['trainloader']))))
            sys.stdout.flush()
            sys.stdout.write(f", training sample {(i+1):.2f}")
            sys.stdout.flush()
        if id == 0:
            print(f'\nMaximum value {torch.max(torch.softmax(outputs,dim=-1),dim=-1)[0][:15]}, \nLocation {torch.argmax(torch.softmax(outputs,dim=-1),dim=-1)[:15]}')
            print('targets',targets[:15])
        lr_stepper.step()
        score_window.append(loss.item())
        scores.append(np.mean(score_window))
        net.eval()
        val_losses = []
        for i, data in enumerate(data_dict['valloader'], 1):
            sys.stdout.write('\r')
            inputs, targets = data.values()
            if training_params['five_card_conversion'] == True:
                inputs = unspool(inputs)
            if training_params['one_hot'] == True:
                inputs = torch.nn.functional.one_hot(inputs)
            inputs = inputs.to(id)
            targets = targets.to(id)
            val_preds = net(inputs)
            val_loss = criterion(val_preds, targets)
            val_losses.append(val_loss.item())
            sys.stdout.write("[%-60s] %d%%" % ('='*(60*(i+1)//len(data_dict['valloader'])), (100*(i+1)//len(data_dict['valloader']))))
            sys.stdout.flush()
            sys.stdout.write(f", validation sample {(i+1):.2f}")
            sys.stdout.flush()
            if i == 100:
                break
        val_window.append(sum(val_losses))
        val_scores.append(np.mean(val_window))
        net.train()
        if id == 0 or id == 'cpu':
            # print('\nguesses',torch.argmax(val_preds,dim=-1)[:15])
            # print('targets',targets[:15])
            print(f"\nTraining loss {np.mean(score_window):.4f}, Val loss {np.mean(val_window):.4f}, Epoch {epoch}")
            print(f"Saving weights to {training_params['load_path']}")
            torch.save(net.state_dict(), training_params['load_path'])
    print('')
    # Save graphs
    # loss_data = [scores,val_scores]
    # loss_labels = ['Training_loss','Validation_loss']
    # plot_data(name,loss_data,loss_labels)
    # # check each hand type
    # if 'y_handtype_indexes' in data_dict:
    #     net.eval()
    #     for handtype in data_dict['y_handtype_indexes'].keys():
    #         mask = data_dict['y_handtype_indexes'][handtype]
    #         inputs = data_dict['valX'][mask]
    #         if training_params['five_card_conversion'] == True:
    #             inputs = unspool(inputs)
    #         if training_params['one_hot'] == True:
    #             inputs = torch.nn.functional.one_hot(inputs)
    #         if inputs.size(0) > 0:
    #             val_preds = net(inputs)
    #             val_loss = criterion(val_preds, data_dict['valY'][mask])
    #             print(f'test performance on {training_params["labels"][handtype]}: {val_loss}')
    #     net.train()
    if torch.cuda.device_count() > 1:
        cleanup()

def train_classification(dataset_params,agent_params,training_params):
    dataset = load_data(dataset_params['data_path'])
    trainloader = return_trainloader(dataset['trainX'],dataset['trainY'],category='classification')
    valloader = return_trainloader(dataset['valX'],dataset['valY'],category='classification')
    data_dict = {
        'trainloader':trainloader,
        'valloader':valloader
        # 'y_handtype_indexes':y_handtype_indexes
    }
    if dataset_params['datatype'] == f'{dt.DataTypes.HANDRANKSFIVE}' or dataset_params['datatype'] == f'{dt.DataTypes.FLATDECK}':
        category_weights = generate_category_weights()
        data_dict['category_weights'] = category_weights
    if dataset_params['datatype'] == f'{dt.DataTypes.SMALLDECK}':
        training_params['frozen'] = True
    print('Data shapes',dataset['trainX'].shape,dataset['trainY'].shape,dataset['valX'].shape,dataset['valY'].shape)
    # dataset['trainY'] = dataset['trainY'].long()
    # dataset['valY'] = dataset['valY'].long()
    # target = dt.Globals.TARGET_SET[dataset_params['datatype']]
    # y_handtype_indexes = return_ylabel_dict(dataset['valX'],dataset['valY'],target)
    # print(f"Target values, Trainset: {np.unique(dataset['trainY'],return_counts=True)}, Valset: {np.unique(dataset['valY'],return_counts=True)}")
    world_size = max(torch.cuda.device_count(),1)
    print(f'World size {world_size}')
    # train_network(0,data_dict,agent_params,training_params)
    mp.spawn(train_network,
        args=(data_dict,agent_params,training_params,),
        nprocs=world_size,
        join=True)

def train_regression(dataset_params,agent_params,training_params):
    dataset = load_data(dataset_params['data_path'])
    trainloader = return_trainloader(dataset['trainX'],dataset['trainY'],category='regression')
    valloader = return_trainloader(dataset['valX'],dataset['valY'],category='regression')

    print('Data shapes',dataset['trainX'].shape,dataset['trainY'].shape,dataset['valX'].shape,dataset['valY'].shape)
    # print(np.unique(dataset['trainY'],return_counts=True),np.unique(dataset['valY'],return_counts=True))
    data_dict = {
        'trainloader':trainloader,
        'valloader':valloader
    }
    world_size = max(torch.cuda.device_count(),1)
    # train_network(0,data_dict,agent_params,training_params)
    mp.spawn(train_network,
        args=(data_dict,agent_params,training_params,),
        nprocs=world_size,
        join=True)

def validate_network(dataset_params,params):
    device = params['network_params']['gpu1']
    examine_params = params['examine_params']
    net = examine_params['network'](params['network_params'])
    load_weights(net,params['load_path'])
    net.to(device)
    net.eval()

    bad_outputs = []
    bad_labels = []
    dataset = load_data(dataset_params['data_path'])
    trainloader = return_trainloader(dataset['trainX'],dataset['trainY'],category='classification')
    # valloader = return_trainloader(dataset['valX'],dataset['valY'],category='classification')
    for i, data in enumerate(trainloader, 1):
        sys.stdout.write('\r')
        # get the inputs; data is a list of [inputs, targets]
        inputs, targets = data.values()
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = net(inputs)
        bool_mask = torch.argmax(outputs,dim=-1) != targets
        if bool_mask.any():
            # print(torch.argmax(torch.softmax(outputs,dim=-1)[bool_mask],dim=-1)[:10])
            # print(targets[bool_mask][:10])
            bad_outputs.append(torch.argmax(torch.softmax(outputs,dim=-1).detach()[bool_mask],dim=-1))
            bad_labels.append(targets[bool_mask])
        sys.stdout.write("[%-60s] %d%%" % ('='*(60*(i+1)//len(trainloader)), (100*(i+1)//len(trainloader))))
        sys.stdout.flush()
        sys.stdout.write(f", training sample {(i+1):.2f}")
        sys.stdout.flush()
    flattened_bad_outputs = [item.item() for sublist in bad_outputs for item in sublist]
    flattened_bad_labels = [item.item() for sublist in bad_labels for item in sublist]
    print(f'\nNumber of incorrect guesses {len(flattened_bad_outputs)}')
    print(f'\nBad guesses {flattened_bad_outputs[:10]}')
    print(f'\nMissed labels {flattened_bad_labels[:10]}')
    return flattened_bad_labels,flattened_bad_labels


def check_network(dataset_params,params):
    messages = {
        dt.LearningCategories.REGRESSION:'Enter in a category [0,1,2] to pick the desired result [-1,0,1]',
        dt.LearningCategories.MULTICLASS_CATEGORIZATION:'Enter in a handtype from 0-8',
        dt.LearningCategories.BINARY_CATEGORIZATION:'Enter in a blocker type from 0-1'
    }
    output_mapping = {
        dt.LearningCategories.MULTICLASS_CATEGORIZATION:F.softmax,
        dt.LearningCategories.REGRESSION:lambda x: x,
        dt.LearningCategories.BINARY_CATEGORIZATION:lambda x: x
    }
    target_mapping = {
        dt.DataTypes.NINECARD:{i:i for i in range(9)},
        dt.DataTypes.FIVECARD:{i:i for i in range(9)},
        dt.DataTypes.HANDRANKSNINE:{i:dt.Globals.HAND_STRENGTH_SAMPLING[i] for i in range(9)},
        dt.DataTypes.HANDRANKSFIVE:{i:dt.Globals.HAND_STRENGTH_SAMPLING[i] for i in range(9)},
        dt.DataTypes.SMALLDECK:{i:i for i in range(1820)},
        dt.DataTypes.FLATDECK:{i:dt.Globals.HAND_STRENGTH_SAMPLING[i] for i in range(9)},
        dt.DataTypes.FLUSH:{i:dt.Globals.HAND_STRENGTH_SAMPLING[i] for i in range(9)},
        dt.DataTypes.THIRTEENCARD:{i:i-1 for i in range(0,3)},
        dt.DataTypes.TENCARD:{i:i-1 for i in range(0,3)},
        dt.DataTypes.PARTIAL:{i:i-1 for i in range(0,3)},
        dt.DataTypes.BLOCKERS:{i:i for i in range(0,2)},
    }
    target = dt.Globals.TARGET_SET[dataset_params['datatype']]
    output_map = output_mapping[dataset_params['learning_category']]
    mapping = target_mapping[dataset_params['datatype']]
    message = messages[dataset_params['learning_category']]
    examine_params = params['examine_params']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = examine_params['network'](params['network_params'])
    load_weights(net,examine_params['load_path'])
    net = net.to(device)
    net.eval()

    dataset = load_data(dataset_params['data_path'])
    valX = dataset['valX']
    valY = dataset['valY']
    y_handtype_indexes = return_ylabel_dict(valX,valY,target)

    while 1:
        human_input = input(message)
        while not human_input.isdigit():
            print('Improper input, must be digit')
            human_input = input(message)
        try:
            if callable(mapping[int(human_input)]) == True:
                category = mapping[int(human_input)]()
            else:
                category = mapping[int(human_input)]
            indicies = y_handtype_indexes[category]
            if len(indicies) > 0:
                print('indicies',indicies.size())
                rand_index = torch.randint(0,indicies.size(0),(1,))
                rand_hand = indicies[rand_index]
                print(f'Evaluating on: {valX[rand_hand]}')
                if torch.cuda.is_available():
                    out = net(torch.tensor(valX[rand_hand]).unsqueeze(0).cuda())
                else:
                    out = net(torch.tensor(valX[rand_hand]).unsqueeze(0))
                print(f'Network output: {output_map(out,dim=-1)}, Maximum value {torch.max(output_map(out,dim=-1))}, Location {torch.argmax(output_map(out,dim=-1))}')
                print(f'Actual category: {valY[rand_hand]}')
            else:
                print('No instances of this, please try again')
        except:
            pass

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=
        """
        Train and evaluate networks on card representations\n\n
        use ```python cards.py -M examine``` to check handtype probabilities
        use ```python cards.py -d random``` to train on predicting winners
        use ```python cards.py``` to train on predicting handtypes
        """)

    parser.add_argument('-d','--datatype',
                        default=dt.DataTypes.HANDRANKSNINE,type=str,
                        metavar=f"[{dt.DataTypes.THIRTEENCARD},{dt.DataTypes.TENCARD},{dt.DataTypes.NINECARD},{dt.DataTypes.FIVECARD},{dt.DataTypes.PARTIAL},{dt.DataTypes.BLOCKERS},{dt.DataTypes.HANDRANKSFIVE},{dt.DataTypes.HANDRANKSNINE}]",
                        help='Which dataset to train on')
    parser.add_argument('-m','--mode',
                        metavar=f"[{dt.Modes.TRAIN}, {dt.Modes.EXAMINE}, {dt.Modes.VALIDATE}]",
                        help='Pick whether you want to train or examine a network',
                        default='train',type=str)
    parser.add_argument('-r','--random',dest='randomize',
                        help='Randomize the dataset. (False -> the data is sorted)',
                        default=True,type=bool)
    parser.add_argument('--encode',metavar=[dt.Encodings.TWO_DIMENSIONAL,dt.Encodings.THREE_DIMENSIONAL],
                        help='Encoding of the cards: 2d -> Hand (4,2). 3d -> Hand (4,13,4)',
                        default=dt.Encodings.TWO_DIMENSIONAL,type=str)
    parser.add_argument('-e','--epochs',
                        help='Number of training epochs',
                        default=1,type=int)
    parser.add_argument('--resume',
                        help='resume training from an earlier run',
                        action='store_true')
    parser.add_argument('-lr',
                        help='resume training from an earlier run',
                        type=float,
                        default=0.003)
    parser.set_defaults(resume=False)


    args = parser.parse_args()

    print('OPTIONS',args)

    learning_category = dt.Globals.DatasetCategories[args.datatype]
    network = NetworkConfig.DataModels[args.datatype]
    network_name = NetworkConfig.DataModels[args.datatype].__name__
    network_path = os.path.join('checkpoints',learning_category,network_name)
    print(f'Loading model {network_path}')

    examine_params = {
        'network':network,
        'load_path':network_path
    }
    dataset_params = {
        'encoding':args.encode,
        'datatype':args.datatype,
        'learning_category':learning_category,
        'data_path':os.path.join('data',dt.Globals.DatasetCategories[args.datatype],args.datatype)
    }
    agent_params = {
        'learning_rate':args.lr,
        'network':NetworkConfig.DataModels[args.datatype],
        'save_dir':'checkpoints',
        'save_path':network_path,
        'load_path':network_path
    }
    network_params = {
        'seed':346,
        'state_space':(13,2),
        'nA':dt.Globals.ACTION_SPACES[args.datatype],
        'channels':13,
        'kernel':2,
        'batchnorm':True,
        'conv_layers':1,
        'load_path':network_path,
        'device' : torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        'gpu1': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        'gpu2': torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    }
    training_params = {
        'resume':args.resume,
        'epochs':args.epochs,
        'five_card_conversion':False,
        'one_hot':False,
        'frozen':False,
        'criterion':NetworkConfig.LossFunctions[dataset_params['learning_category']],
        'network': network,
        'load_path':network_path,
        'labels':dt.Globals.LABEL_DICT[args.datatype],
        'gpu1': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        'gpu2': torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
    }
    multitrain_params = {
        'conversion_list':[False],#,False],#[,True],
        'onehot_list':[False],#,False],#[,True]
        'networks':[HandClassificationV2],#HandClassification,HandClassificationV2,HandClassificationV3,HandClassificationV4],#,
    }
    agent_params['network_params'] = network_params
    agent_params['examine_params'] = examine_params
    agent_params['multitrain_params'] = multitrain_params
    tic = time.time()
    if args.mode == dt.Modes.EXAMINE:
        check_network(dataset_params,agent_params)
    elif args.mode == dt.Modes.VALIDATE:
        validate_network(dataset_params,agent_params)
    elif args.mode == dt.Modes.MULTITRAIN:
        # load config
        with open(os.path.join(os.getcwd(),'network_configs.yaml')) as file:
            yaml_configs = yaml.load(file,Loader=yaml.FullLoader)
            net_configs = yaml_configs[args.datatype]
            for k,v in net_configs.items():
                print(f'Training {k}, with params {v}')
                network_params['hidden_dims'] = v['hidden_dims']
                network_params['board_dims'] = v['board_dims']
                network_params['hand_dims'] = v['hand_dims']
                agent_params['network_params'] = network_params
                train_classification(dataset_params,agent_params,training_params)
                bad_guesses,missed_labels = validate_network(dataset_params,agent_params)
                client = MongoClient('localhost', 27017,maxPoolSize=10000)
                db = client['poker']
                state_json = {'epochs_trained':args.epochs,'network_arch':v,'network_name':k,'bad_guesses':bad_guesses,'missed_labels':missed_labels,'num_missed':len(bad_guesses)}
                db['network_results'].insert_one(state_json)
                client.close()
    elif args.mode == dt.Modes.TRAIN:
        print(f'Evaluating {network_name} on {args.datatype}, {dataset_params["learning_category"]}')
        if learning_category == dt.LearningCategories.MULTICLASS_CATEGORIZATION or learning_category == dt.LearningCategories.BINARY_CATEGORIZATION:
            train_classification(dataset_params,agent_params,training_params)
        elif learning_category == dt.LearningCategories.REGRESSION:
            train_regression(dataset_params,agent_params,training_params)
        else:
            raise ValueError(f'{args.datatype} datatype not understood')
    else:
        raise ValueError(f'{args.mode} Mode not understood')
    toc = time.time()
    print(f'Evaluation took {(toc-tic)/60} minutes')
    