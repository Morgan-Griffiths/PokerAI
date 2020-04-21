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
from agents.agent import CardAgent
from data_loader import return_dataloader
import hand_recognition.datatypes as dt
from models.networks import *
from models.network_config import NetworkConfig
from hand_recognition.data_utils import unpack_nparrays,load_data,return_handtype_dict,load_handtypes,return_handtype_data_shapes

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


def multi_train(agent_params:list,training_params:dict,dataset_params:dict):
    data = load_data(dataset_params['data_path'])
    scores = []
    for agent_param in agent_params:
        print(f'New Agent params: {agent_param}')
        agent = CardAgent(agent_param)
        result = train_agent(data,agent,training_params)
        scores.append(result)
    return scores

def train_agent(data:dict,agent,params):
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
    results = multi_train(agent_param_list,training_params,dataset_params)
    graph_networks(results)

def train_network(data_dict,agent_params,training_params):
    net = training_params['network'](agent_params['network_params'])
    criterion = training_params['criterion']()
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
    if 'y_handtype_indexes' in data_dict:
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
    torch.save(net.state_dict(), training_params['save_path'])

def train_classification(dataset_params,agent_params,training_params):
    dataset = load_handtypes(dataset_params['data_path'])
    trainset = dataset['train']
    testset = dataset['test']
    train_shape,train_batch = return_handtype_data_shapes(trainset)
    test_shape,test_batch = return_handtype_data_shapes(testset)

    print(f'train_shape {train_shape}, train_batch {train_batch}')
    print(f'test_shape {test_shape}, test_batch {test_batch}')

    trainX,trainY = unpack_nparrays(train_shape,train_batch,trainset)
    valX,valY = unpack_nparrays(test_shape,test_batch,testset)
    y_handtype_indexes = return_handtype_dict(valX,valY)
    trainloader = return_dataloader(trainX,trainY)

    print(trainX.size(),trainY.size(),valX.size(),valY.size())
    print(np.unique(trainY,return_counts=True),np.unique(valY,return_counts=True))

    data_dict = {
        'trainloader':trainloader,
        'valX':valX,
        'valY':valY,
        'y_handtype_indexes':y_handtype_indexes
    }
    train_network(data_dict,agent_params,training_params)

def train_regression(dataset_params,agent_params,training_params):
    dataset = load_data(dataset_params['data_path'])
    trainloader = return_dataloader(dataset['trainX'],dataset['trainY'])
    print(np.unique(dataset['trainY'],return_counts=True),np.unique(dataset['valY'],return_counts=True))
    data_dict = {
        'trainloader':trainloader,
        'valX':dataset['valX'],
        'valY':dataset['valY']
    }
    train_network(data_dict,agent_params,training_params)

def check_network(dataset_params,params):
    messages = {
        dt.LearningCategories.REGRESSION:'Enter in a category [0,1,2] to pick the desired result [-1,0,1]',
        dt.LearningCategories.MULTICLASS_CATEGORIZATION:'Enter in a handtype from 0-8'
    }
    target_mapping = {
        dt.LearningCategories.MULTICLASS_CATEGORIZATION:{i:i for i in range(9)},
        dt.LearningCategories.REGRESSION:{i:i-1 for i in range(0,3)}
    }
    output_mapping = {
        dt.LearningCategories.MULTICLASS_CATEGORIZATION:F.softmax,
        dt.LearningCategories.REGRESSION:lambda x: x
    }
    output_map = output_mapping[dataset_params['learning_category']]
    mapping = target_mapping[dataset_params['learning_category']]
    message = messages[dataset_params['learning_category']]
    examine_params = params['examine_params']
    net = examine_params['network'](params['network_params'])
    net.load_state_dict(torch.load(examine_params['load_path']))
    net.eval()
    if dataset_params['learning_category'] == dt.LearningCategories.MULTICLASS_CATEGORIZATION:
        dataset = load_handtypes(dataset_params['data_path'])
        testset = dataset['test']
        test_shape,test_batch = return_handtype_data_shapes(testset)
        valX,valY = unpack_nparrays(test_shape,test_batch,testset)
        y_handtype_indexes = return_handtype_dict(valX,valY)
    elif dataset_params['learning_category'] == dt.LearningCategories.REGRESSION:
        dataset = load_data(dataset_params['data_path'])
        target_dict = {-1:-1,0:0,1:1}
        valX = dataset['valX']
        valY = dataset['valY']
        y_handtype_indexes = return_handtype_dict(valX,valY,target_dict)
    else:
        raise ValueError(f'Learning category {dataset_params["learning_category"]} not supported')
    while 1:
        human_input = input(message)
        while not human_input.isdigit():
            print('Improper input, must be digit')
            human_input = input(message)
        category = mapping[int(human_input)]
        indicies = y_handtype_indexes[category]
        print('indicies',indicies.size())
        rand_index = torch.randint(0,indicies.size(0),(1,))
        rand_hand = indicies[rand_index]
        print(f'Evaluating on: {valX[rand_hand]}')
        out = net(valX[rand_hand])
        print(f'Network output: {output_map(out)}')
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
                        metavar=f"[{dt.DataTypes.THIRTEENCARD},{dt.DataTypes.TENCARD},{dt.DataTypes.NINECARD},{dt.DataTypes.FIVECARD},{dt.DataTypes.PARTIAL},{dt.DataTypes.BLOCKERS}]",
                        help='Which dataset to train on')
    parser.add_argument('-M','--mode',
                        metavar=f"[{dt.Modes.TRAIN}, {dt.Modes.EXAMINE}]",
                        help='Pick whether you want to train,build or examine a network',
                        default='train',type=str)
    parser.add_argument('-r','--random',dest='randomize',
                        help='Randomize the dataset. (False -> the data is sorted)',
                        default=True,type=bool)
    parser.add_argument('--encode',metavar=[dt.Encodings.TWO_DIMENSIONAL,dt.Encodings.THREE_DIMENSIONAL],
                        help='Encoding of the cards: 2d -> Hand (4,2). 3d -> Hand (4,13,4)',
                        default=dt.Encodings.TWO_DIMENSIONAL,type=str)

    args = parser.parse_args()

    print('OPTIONS',args)

    learning_category = dt.Globals.DatasetCategories[args.datatype]
    network = NetworkConfig.DataModels[args.datatype]
    network_name = NetworkConfig.DataModels[args.datatype].__name__
    network_path = os.path.join('checkpoints',learning_category,network_name)

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
        'learning_rate':2e-3,
        'network':NetworkConfig.DataModels[args.datatype],
        'save_dir':'checkpoints',
        'save_path':network_path,
        'load_path':network_path
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
        'epochs':10,
        'five_card_conversion':False,
        'one_hot':False,
        'criterion':NetworkConfig.LossFunctions[dataset_params['learning_category']],
        'network': network,
        'save_path':network_path
    }
    multitrain_params = {
        'conversion_list':[False],#,False],#[,True],
        'onehot_list':[False],#,False],#[,True]
        'networks':[HandClassificationV2],#HandClassification,HandClassificationV2,HandClassificationV3,HandClassificationV4],#,
    }
    agent_params['network_params'] = network_params
    agent_params['examine_params'] = examine_params
    agent_params['multitrain_params'] = multitrain_params
    # compute_baseline()
    if args.mode == dt.Modes.EXAMINE:
        check_network(dataset_params,agent_params)
    elif args.mode == dt.Modes.TRAIN:
        print(f'Evaluating {network_name} on {args.datatype}, {dataset_params["learning_category"]}')
        if learning_category == dt.LearningCategories.MULTICLASS_CATEGORIZATION:
            train_classification(dataset_params,agent_params,training_params)
        elif learning_category == dt.LearningCategories.BINARY_CATEGORIZATION:
            pass
        elif learning_category == dt.LearningCategories.REGRESSION:
            train_regression(dataset_params,agent_params,training_params)
        else:
            raise ValueError(f'{args.datatype} datatype not understood')
    else:
        raise ValueError(f'{args.mode} Mode not understood')
    