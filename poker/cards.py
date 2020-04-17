import torch.nn.functional as F
import torch
import numpy as np
import os
import sys
import copy
from torch import optim
from random import shuffle
from collections import deque
from itertools import combinations

from visualize import plot_data
from models.networks import *
from agents.agent import CardAgent
from data_loader import return_dataloader
import hand_recognition.datatypes as dt
from hand_recognition.data_utils import unpack_nparrays,load_data,save_data,return_handtype_dict
from hand_recognition.build_data import CardDataset

"""
Creating a hand dataset for training and evaluating networks.
Full deck
Omaha
"""

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
            combined[:,i,:,:] = torch.cat((torch.stack(hcombo),torch.stack(bcombo)),dim=0).permute(1,0,2)
            i += 1
    return combined


def multi_train(agent_params:list,training_params:dict):
    data = load_data()
    scores = []
    for agent_param in agent_params:
        print(f'New Agent params: {agent_param}')
        agent = CardAgent(agent_param)
        result = train_network(data,agent,training_params)
        scores.append(result)
    return scores

def train_network(data,agent,params):
    train_means,val_means = [],[]
    train_scores = deque(maxlen=50)
    val_scores = deque(maxlen=50)
    for i in range(1,101):
        preds = agent(data['trainX'])
        loss = agent.backward(preds,data['trainY'])
        val_preds = agent.predict(data['valX'])
        val_loss = F.smooth_l1_loss(val_preds,data['valY']).item()
        train_scores.append(loss)
        val_scores.append(val_loss)
        train_means.append(np.mean(train_scores))
        val_means.append(np.mean(val_scores))
        if i % 10 == 0:
            print(f'Episode: {i}, loss: {np.mean(train_scores)}, val loss: {np.mean(val_scores)}')
    agent.save_weights()
    return {'train_scores':train_means,'val_scores':val_means,'name':agent.agent_name}


def graph_networks(results:list):
    labels = ['Training Loss','Val Loss']
    for result in results:
        train_loss = result['train_scores']
        val_loss = result['val_scores']
        name = result['name']
        plot_data(name,[train_loss,val_loss],labels)


def evaluate_random_hands(dataset_params,agent_params,training_params):
    Agent_paths = ['Conv(kernel2)',
                'Conv(kernel2)_layer3',
                'Conv(kernel13)',
                'Conv(kernel13)_layer3',
                'Embedding_FC']
    Agent_networks = [CardClassification,
                    CardClassification,
                    CardClassification,
                    CardClassification,
                    CardClassificationV2]
    permute_input = [False,False,True,True,False]
    channel_list = [[13],[13,64,64],[2],[2,64,64],0]
    kernel_list = [[2],[2,1,1],[13],[13,1,1],0]
    layer_list = [1,3,1,3,0]
    # Variations
    agent_param_list = []
    for i in range(0,len(Agent_networks)):
        agent_params = copy.deepcopy(agent_params)
        agent_params['network'] = Agent_networks[i]
        agent_params['network_params']['kernel'] = kernel_list[i]
        agent_params['network_params']['channels'] = channel_list[i]
        agent_params['network_params']['conv_layers'] = layer_list[i]
        agent_params['network_params']['permute'] = permute_input[i]
        agent_params['save_path'] = Agent_paths[i]
        agent_params['load_path'] = Agent_paths[i]
        agent_param_list.append(agent_params)
    results = multi_train(agent_param_list,training_params)
    graph_networks(results)

def evaluate_handtypes(dataset_params,agent_params,training_params):
    # Load data
    train_shape = (45000,9,2)
    train_batch = 5000
    test_shape = (9000,9,2)
    test_batch = 1000
    train_data = load_data('data/hand_types/train')
    trainX,trainY = unpack_nparrays(train_shape,train_batch,train_data)
    val_data = load_data('data/hand_types/test')
    valX,valY = unpack_nparrays(test_shape,test_batch,val_data)
    y_handtype_indexes = return_handtype_dict(valX,valY)
    trainloader = return_dataloader(trainX,trainY)
    data_dict = {
        'trainloader':trainloader,
        'valX':valX,
        'valY':valY,
        'y_handtype_indexes':y_handtype_indexes
    }
    # Loop over all network choices
    for net_idx,network in enumerate(training_params['networks']):
        training_params['five_card_conversion'] = training_params['conversion_list'][net_idx]
        train_classification(data_dict,network,agent_params,training_params)

def train_classification(data_dict,network,agent_params,training_params):
    print(f'Training {network.__name__} network')
    net = network(agent_params['network_params'])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.003)
    scores = []
    val_scores = []
    score_window = deque(maxlen=100)
    val_window = deque(maxlen=100)
    for epoch in range(training_params['epochs']):
        for i, data in enumerate(data_dict['trainloader'], 1):
            # get the inputs; data is a list of [inputs, targets]
            inputs, targets = data.values()
            # zero the parameter gradients
            optimizer.zero_grad()
            # unspool hand into 60,5 combos
            if training_params['five_card_conversion'] == True:
                inputs = unspool(inputs)
            if training_params['one_hot'] == True:
                inputs = torch.nn.functional.one_hot(inputs)
            outputs = net(inputs)
            # print(type(targets),targets,targets.size())
            # print(outputs,outputs.size())
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            score_window.append(loss.item())
            scores.append(np.mean(score_window))
            net.eval()
            val_inputs = data_dict['valX']
            if training_params['five_card_conversion'] == True:
                val_inputs = unspool(val_inputs)
            if training_params['one_hot'] == True:
                inputs = torch.nn.functional.one_hot(val_inputs)
            val_preds = net(val_inputs)
            val_loss = criterion(val_preds, data_dict['valY'])
            val_window.append(val_loss.item())
            val_scores.append(np.mean(val_window))
            net.train()
        print(f'Episode {epoch} loss {np.mean(score_window)}')
    # Save graphs
    loss_data = [scores,val_scores]
    loss_labels = ['Training_loss','Validation_loss']
    plot_data(f'{network.__name__}_Handtype_categorization',loss_data,loss_labels)
    # check each hand type
    net.eval()
    for handtype in data_dict['y_handtype_indexes'].keys():
        mask = data_dict['y_handtype_indexes'][handtype]
        inputs = data_dict['valX'][mask]
        if training_params['five_card_conversion'] == True:
            inputs = unspool(inputs)
        if training_params['one_hot'] == True:
            inputs = torch.nn.functional.one_hot(inputs)
        val_preds = net(inputs)
        val_loss = criterion(val_preds, data_dict['valY'][mask])
        print(f'test performance on {dt.Globals.HAND_TYPE_DICT[handtype]}: {val_loss}')
    net.train()
    torch.save(net.state_dict(), f'checkpoints/hand_categorization/{network.__name__}')


def evaluate_fivecard(dataset_params,agent_params,training_params):
    # Load data
    train_shape = (50000,5,2)
    train_batch = 5000
    test_shape = (10000,5,2)
    test_batch = 1000
    fivecard_data = load_data('data/fivecard')
    print(fivecard_data.keys())
    valX,valY,trainX,trainY = fivecard_data.values()
    trainY = trainY.squeeze(-1).long()
    valY = valY.squeeze(-1).long()
    print(trainX.size(),trainY.size(),valX.size(),valY.size())
    print(np.unique(trainY,return_counts=True),np.unique(valY,return_counts=True))
    y_handtype_indexes = return_handtype_dict(valX,valY)
    trainloader = return_dataloader(trainX,trainY)
    data_dict = {
        'trainloader':trainloader,
        'valX':valX,
        'valY':valY,
        'y_handtype_indexes':y_handtype_indexes
    }
    network = FiveCardClassification
    train_classification(data_dict,network,agent_params,training_params)

def compute_baseline():
    """
    predicts most common class always for a baseline comparison.
    """
    print('Computing baseline, predict most common case')
    fivecard_data = load_data('data/fivecard')
    valX,valY,trainX,trainY = fivecard_data.values()
    trainY = trainY.squeeze(-1).long()
    print('valX,valY,trainX,trainY',trainX.size(),trainY.size())
    trainloader = return_dataloader(trainX,trainY)
    uniques,counts = np.unique(trainY,return_counts=True)
    print(uniques,counts)
    most_common_category = np.argmax(counts)
    print('most_common_category',most_common_category)
    criterion =  nn.MultiLabelSoftMarginLoss()
    score_window = deque(maxlen=100)
    scores = []
    num_categories = len(counts)
    pred = torch.zeros(num_categories)
    pred[most_common_category] = 1
    for i, data in enumerate(trainloader, 1): 
        inputs,targets = data.values()
        M = inputs.size(0)
        predictions = pred.repeat(M).view(M,num_categories).float()
        labels = torch.zeros((M,num_categories))
        labels[targets] = 1
        loss = criterion(predictions,labels)
        score_window.append(loss.sum())
        scores.append(np.mean(score_window))
    # print('baseline',scores)
    print('baseline',np.mean(score_window))

def check_network(params):
    examine_params = params['examine_params']
    net = examine_params['network'](params['network_params'])
    net.load_state_dict(torch.load(examine_params['load_path']))
    net.eval()
    test_shape = (9000,5,2)#(9000,9,2)
    test_batch = 1000
    val_data = load_data('data/fivecard')
    valX,valY = unpack_nparrays(test_shape,test_batch,val_data)
    y_handtype_indexes = return_handtype_dict(valX,valY)
    while 1:
        human_input = input('Enter in a handtype from 0-8')
        while not human_input.isdigit():
            print('Improper input, must be digit')
            human_input = input('Enter in a handtype from 0-8')
        handtype = int(human_input)
        type_index = y_handtype_indexes[handtype]
        indicies = (type_index != 0).nonzero()
        rand_hand = torch.randint(torch.min(indicies),torch.max(indicies),(1,))
        print(f'Evaluating on: {valX[rand_hand]}')
        out = net(valX[rand_hand])
        print(f'Network output: {F.softmax(out,dim=-1)}')
        print(f'Actual category: {valY[rand_hand]}')

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
                        default='handtype',type=str,
                        metavar=f"[{dt.DataTypes.HANDTYPE},{dt.DataTypes.RANDOM},{dt.DataTypes.FIVECARD}]",
                        help='Which dataset to train or build')
    parser.add_argument('-bi','--builditerations',
                        dest='build_iterations',
                        help='Number of building iterations',
                        default=600000,type=int)
    parser.add_argument('-m','--maxlen',
                        help='maxlen of data deques for building data',
                        default=1000,type=int)
    parser.add_argument('-M','--mode',
                        metavar=f"[{dt.Modes.TRAIN},{dt.Modes.BUILD},{dt.Modes.EXAMINE}]",
                        help='Pick whether you want to train,build or examine a network',
                        default='train',type=str)
    parser.add_argument('-O','--datapath',
                        help='Local path to save data',
                        default='data/hand_types/test',type=str)
    parser.add_argument('--trainsize',
                        help='Size of train set',
                        default=50000,type=int)
    parser.add_argument('--valsize',
                        help='Size of test set',
                        default=10000,type=int)

    args = parser.parse_args()

    print('OPTIONS',args)

    examine_params = {
        'network':FiveCardClassification,
        'load_path':os.path.join('checkpoints/hand_categorization','FiveCardClassification')
    }
    dataset_params = {
        'training_set_size':args.trainsize,
        'val_set_size':args.valsize,
        'encoding':'2d',
        'maxlen':args.maxlen,
        'iterations':args.build_iterations,
        'save_path':args.datapath,
        'datatype':args.datatype
    }
    agent_params = {
        'learning_rate':2e-3,
        'network':CardClassification,
        'save_dir':'checkpoints',
        'save_path':'Conv(kernel2)',
        'load_path':'Conv(kernel2)'
    }
    network_params = {
        'seed':346,
        'state_space':(13,2),
        'nA':9,
        'channels':13,
        'kernel':2,
        'batchnorm':True,
        'conv_layers':1,
    }
    training_params = {
        'epochs':3,
        'networks':[HandClassificationV2],#HandClassification,HandClassificationV2,HandClassificationV3,HandClassificationV4],#,
        'five_card_conversion':False,
        'one_hot':False,
        'conversion_list':[False],#,False],#[,True],
        'onehot_list':[False]#,False],#[,True]
    }
    agent_params['network_params'] = network_params
    agent_params['examine_params'] = examine_params

    # dataset = CardDataset(dataset_params)
    # while 1:
    #     handtype = input('Enter in int 0-8 to pick handtype')
    #     hand = dataset.create_handtypes(int(handtype))
    #     print(f'Hand {hand}, Category {handtype}')
    # compute_baseline()
    if args.mode == dt.Modes.EXAMINE:
        check_network(agent_params)
    elif args.mode == dt.Modes.BUILD:
        print(f'Building {args.datatype} dataset')
        dataset = CardDataset(dataset_params)
        if args.datatype == dt.DataTypes.HANDTYPE or args.datatype == dt.DataTypes.FIVECARD:
            dataset.build_hand_classes(dataset_params)
        elif args.datatype == dt.DataTypes.RANDOM:
            trainX,trainY,valX,valY = dataset.generate_dataset(dataset_params)
            save_data(trainX,trainY,valX,valY,dataset_params)
        else:
            raise ValueError(f'{args.datatype} datatype not understood')
    elif args.mode == dt.Modes.TRAIN:
        print(f'Evaluating networks on {args.datatype}')
        if args.datatype == dt.DataTypes.HANDTYPE:
            evaluate_handtypes(dataset_params,agent_params,training_params)
        elif args.datatype == dt.DataTypes.RANDOM:
            evaluate_random_hands(dataset_params,agent_params,training_params)
        elif args.datatype == dt.DataTypes.FIVECARD:
            evaluate_fivecard(dataset_params,agent_params,training_params)
        else:
            raise ValueError(f'{args.datatype} datatype not understood')
    else:
        raise ValueError(f'{args.mode} Mode not understood')
    