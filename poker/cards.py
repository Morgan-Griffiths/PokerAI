import torch.nn.functional as F
import torch
import numpy as np
import timeit
import os
import copy
from torch import optim
from random import shuffle
from collections import deque
from itertools import combinations

from visualize import plot_data
from cardlib import encode,decode,winner,hand_rank
from models.networks import *
from agents.agent import CardAgent
from card_utils import to_2d,suits_to_str,convert_numpy_to_rust,convert_numpy_to_2d
from data_loader import return_dataloader
from utils import torch_where

"""
Creating a hand dataset for training and evaluating networks.
Full deck
Omaha
"""

hand_type_dict = {
            0:'Straight_flush',
            1:'Four_of_a_kind',
            2:'Full_house',
            3:'Flush',
            4:'Straight',
            5:'Three_of_a_kind',
            6:'Two_pair',
            7:'One_pair',
            8:'High_card'
        }

hand_type_file_dict = {'Hand_type_'+v:k for k,v in hand_type_dict.items()}

class CardDataset(object):
    def __init__(self,params):
        self.deck = np.arange(52)
        self.params = params

    def generate_hands(self,iterations,encoding):
        X,y = [],[]
        for i in range(iterations):
            cards = np.random.choice(self.deck,13,replace=False)
            rust_cards = convert_numpy_to_rust(cards)
            encoded_cards = [encode(c) for c in rust_cards]
            hand1 = encoded_cards[:4]
            hand2 = encoded_cards[4:8]
            board = encoded_cards[8:]
            result = winner(hand1,hand2,board)
            if encoding == '2d':
                cards2d = convert_numpy_to_2d(cards)
                X.append(cards2d)
            else:
                X.append(cards)
            y.append(result)
            # print('result',result)
        X = np.stack(X)
        y = np.stack(y)[:,None]
        return X,y

    def build_hand_classes(self,params):
        """
        |Hand Value|Unique|Distinct|
        |Straight Flush |40      |10|
        |Four of a Kind |624     |156|
        |Full Houses    |3744    |156|
        |Flush          |5108    |1277|
        |Straight       |10200   |10|
        |Three of a Kind|54912   |858|
        |Two Pair       |123552  |858|
        |One Pair       |1098240 |2860|
        |High Card      |1302540 |1277|
        |TOTAL          |2598960 |7462|
        """
        
        hand_strengths = {i:deque(maxlen=params['maxlen']) for i in range(1,10)}
        for _ in range(600000):
            cards = np.random.choice(self.deck,13,replace=False)
            rust_cards = convert_numpy_to_rust(cards)
            numpy_cards = convert_numpy_to_2d(cards)
            hand1 = rust_cards[:4]
            hand2 = rust_cards[4:8]
            board = rust_cards[8:]
            en_hand1 = [encode(c) for c in hand1]
            en_hand2 = [encode(c) for c in hand2]
            en_board = [encode(c) for c in board]
            rank = hand_rank(en_hand1,en_board)
            hand_type = CardDataset.find_strength(rank)
            hand_strengths[hand_type].append(numpy_cards[:4]+numpy_cards[8:])
            rank = hand_rank(en_hand2,en_board)
            hand_type = CardDataset.find_strength(rank)
            hand_strengths[hand_type].append(numpy_cards[4:8]+numpy_cards[8:])
        [print(len(hand_strengths[i])) for i in range(1,10)]
        for i in range(1,10):
            np.save(os.path.join(params['save_path'],f'Hand_type_{hand_type_dict[i]}'),hand_strengths[i])

    @staticmethod
    def find_strength(strength):
        # 7462-6185 High card
        # 6185-3325 Pair
        # 3325-2467 2Pair
        # 2467-1609 Trips
        # 1609-1599  Stright
        # 1599-322 Flush
        # 322-166  FH
        # 166-10 Quads
        # 10-0 Str8 flush
        if strength > 6185:
            return 8
        if strength > 3325:
            return 7
        if strength > 2467:
            return 6
        if strength > 1609:
            return 5
        if strength > 1599:
            return 4
        if strength > 322:
            return 3
        if strength > 166:
            return 2
        if strength > 10:
            return 1
        return 0


    @staticmethod
    def to_torch(inputs:list):
        return [torch.Tensor(x) for x in inputs]

    def generate_dataset(self,params):
        """
        Hands in test set may or may not match hands in training set.
        """
        trainX,trainY = self.generate_hands(params['training_set_size'],params['encoding'])
        valX,valY = self.generate_hands(params['test_set_size'],params['encoding'])
        trainX,trainY,valX,valY = CardDataset.to_torch([trainX,trainY,valX,valY])
        print(f'trainX: {trainX.shape}, trainY {trainY.shape}, valX {valX.shape}, valY {valY.shape}')
        return trainX,trainY,valX,valY

def unpack_nparrays(shape,batch,data):
    X = np.zeros(shape)
    Y = np.zeros(shape[0])
    i = 0
    j = 0
    for k,v in data.items():
        Y[i*batch:(i+1)*batch] = hand_type_file_dict[k]
        for hand in v:
            X[j] = np.stack(hand)
            j += 1
        i += 1
    print('Numpy data uniques and counts ',np.lib.arraysetops.unique(Y,return_counts=True))
    return torch.tensor(X),torch.tensor(Y).long()

def load_data(dir_path='data/predict_winner'):
    data = {}
    for f in os.listdir(dir_path):
        if f != '.DS_store':
            name = os.path.splitext(f)[0]
            data[name] = torch.Tensor(np.load(os.path.join(dir_path,f)))
    return data

def save_data(params,dir_path='data/predict_winner'):
    dataset = CardDataset(params)
    trainX,trainY,valX,valY = dataset.generate_dataset(params)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    np.save('data/predict_winner/trainX',trainX)
    np.save('data/predict_winner/trainY',trainY)
    np.save('data/predict_winner/valX',valX)
    np.save('data/predict_winner/valY',valY)

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

def multi_train(agent_params:list,training_params:dict):
    data = load_data()
    scores = []
    for agent_param in agent_params:
        print(f'New Agent params: {agent_param}')
        agent = CardAgent(agent_param)
        result = train_network(data,agent,training_params)
        scores.append(result)
    return scores

def graph_networks(results:list):
    labels = ['Training Loss','Val Loss']
    for result in results:
        train_loss = result['train_scores']
        val_loss = result['val_scores']
        name = result['name']
        plot_data(name,[train_loss,val_loss],labels)

def return_handtype_dict(X:torch.tensor,y:torch.tensor):
    type_dict = {}
    for key in hand_type_dict.keys():
        if key == 0:
            mask = torch.zeros_like(y)
            mask[(y == 0).nonzero().unsqueeze(0)] = 1
            type_dict[key] = mask
        else:
            type_dict[key] = torch_where(y==key,y)
        assert(torch.max(type_dict[key]).item() == 1)
    return type_dict

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
    # Loop over all network choices
    for net_idx,network in enumerate(training_params['networks']):
        print(f'Training {network.__name__} network')
        net = network(agent_params['network_params'])
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.003)
        scores = []
        val_scores = []
        score_window = deque(maxlen=100)
        val_window = deque(maxlen=100)
        for epoch in range(20):
            for i, data in enumerate(trainloader, 1):
                # get the inputs; data is a list of [inputs, targets]
                inputs, targets = data.values()
                # zero the parameter gradients
                optimizer.zero_grad()
                # unspool hand into 60,5 combos
                if training_params['five_card_conversion'][net_idx] == True:
                    inputs = unspool(inputs)
                outputs = net(inputs)
                # print(type(targets),targets)
                # print(outputs.size(),targets.size())
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                score_window.append(loss.item())
                scores.append(np.mean(score_window))
                net.eval()
                if training_params['five_card_conversion'][net_idx] == True:
                    val_inputs = unspool(valX)
                else:
                    val_inputs = valX
                val_preds = net(val_inputs)
                val_loss = criterion(val_preds, valY)
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
        for handtype in y_handtype_indexes.keys():
            mask = y_handtype_indexes[handtype]
            if training_params['five_card_conversion'][net_idx] == True:
                inputs = unspool(valX[mask])
            else:
                inputs = valX[mask]
            val_preds = net(inputs)
            val_loss = criterion(val_preds, valY[mask])
            print(f'test performance on {hand_type_dict[handtype]}: {val_loss}')
        net.train()
        torch.save(net.state_dict(), f'checkpoints/hand_categorization/{network.__name__}')

def check_network(params):
    examine_params = params['examine_params']
    net = examine_params['network'](params['network_params'])
    net.load_state_dict(torch.load(examine_params['load_path']))
    net.eval()
    test_shape = (9000,9,2)
    test_batch = 1000
    val_data = load_data('data/hand_types/test')
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
        print(f'Network output: {F.softmax(out)}')
        print(f'Actual category: {valY[rand_hand]}')

def compute_baseline():
    test_shape = (9000,9,2)
    test_batch = 1000
    val_data = load_data('data/hand_types/test')
    valX,valY = unpack_nparrays(test_shape,test_batch,val_data)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=
        """
        Train and evaluate networks on card representations\n\n
        use ```python cards.py --examine True``` to check handtype probabilities
        """)

    parser.add_argument('-d','--datatype',
                        default='handtype',type=str,
                        metavar="[handtype,random]",
                        help='Which dataset to train or build')
    parser.add_argument('-b','--build',
                        dest='build',
                        default=False,type=bool,
                        help='To build a dataset or not')
    parser.add_argument('-t','--train',
                        default=True,type=bool,
                        help='To train on a dataset or not')
    parser.add_argument('--examine',
                        default=False,type=bool,
                        help='To train on a dataset or not')
    parser.add_argument('-bi','--builditerations',
                        dest='build_iterations',
                        help='Number of building iterations',
                        default=600000,type=int)
    parser.add_argument('-m','--maxlen',
                        help='maxlen of data deques for building data',
                        default=1000,type=int)
    parser.add_argument('-O','--datapath',
                        help='Local path to save data',
                        default='data/hand_types/train',type=str)

    args = parser.parse_args()

    print('OPTIONS',args)

    examine_params = {
        'network':HandClassificationV2,
        'load_path':os.path.join('checkpoints/hand_categorization','HandClassificationV2')
    }
    dataset_params = {
        'training_set_size':50000,
        'val_set_size':10000,
        'encoding':'2d',
        'handtype_dir':'data/hand_types',
        'predict_winner_dir':'data/predict_winner',
        'maxlen':args.maxlen,
        'iterations':args.build_iterations,
        'save_path':args.datapath

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
        'episodes':5,
        'networks':[HandClassificationV2],#HandClassification,HandClassificationV2,HandClassificationV3,HandClassificationV4],#,
        'five_card_conversion':[False]#,False],#[,True]
    }
    agent_params['network_params'] = network_params
    agent_params['examine_params'] = examine_params
    
    if args.examine == True:
        check_network(agent_params)
    else:
        if args.build == True:
            if args.datatype == 'handtype':
                dataset = CardDataset(dataset_params)
                dataset.build_hand_classes(dataset_params)
            elif args.datatype == 'random':
                save_data(dataset_params)
            else:
                raise ValueError(f'Datatype not recognized {args.datatype}')
        elif args.train == True:
            print(f'Evaluating networks on {args.datatype}')
            if args.datatype == 'random':
                evaluate_random_hands(dataset_params,agent_params,training_params)
            elif args.datatype == 'handtype':
                evaluate_handtypes(dataset_params,agent_params,training_params)
            else:
                raise ValueError(f'{args.datatype} datatype not understood')