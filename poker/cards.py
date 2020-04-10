
from cardlib import encode,decode,winner
from visualize import plot_data
from random import shuffle
import numpy as np
from torch import optim
from collections import deque
import torch
import torch.nn.functional as F
import timeit
from models.networks import CardClassification,CardClassificationV2,CardClassificationV3
from agents.agent import CardAgent
import os
import copy

from card_utils import to_2d,suits_to_str,convert_numpy_to_rust,convert_numpy_to_2d

"""
Creating a hand dataset for training and evaluating networks.
Full deck
Omaha
"""

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
            # print('raw',hand1,hand2,board)
            # hand1 = convert_numpy_to_rust(hand1)
            # hand2 = convert_numpy_to_rust(hand2)
            # board = convert_numpy_to_rust(board)
            # print(f'converted \nhand1 {rust_cards[:4]}, \nhand2 {rust_cards[4:8]}, \nboard {rust_cards[8:]}')
            # en_h1 = [encode(c) for c in hand1]
            # en_h2 = [encode(c) for c in hand2]
            # en_board = [encode(c) for c in board]
            # print('encoded',en_h1,en_h2,en_board)
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

def load_data():
    trainX = np.load('data/trainX.npy')
    trainY = np.load('data/trainY.npy')
    valX = np.load('data/valX.npy')
    valY = np.load('data/valY.npy')

    #to torch
    trainX,trainY,valX,valY = CardDataset.to_torch([trainX,trainY,valX,valY])

    data = {
        'trainX':trainX,
        'trainY':trainY,
        'valX':valX,
        'valY':valY,
    }
    
    print(f'trainX: {trainX.shape}, trainY {trainY.shape}, valX {valX.shape}, valY {valY.shape}')
    return data

def save_data(params):
    dataset = CardDataset(params)
    trainX,trainY,valX,valY = dataset.generate_dataset(params)
    np.save('data/trainX',trainX)
    np.save('data/trainY',trainY)
    np.save('data/valX',valX)
    np.save('data/valY',valY)

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

def compare_networks(agent_params:list,training_params:dict):
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

if __name__ == "__main__":
    params = {
        'training_set_size':50000,
        'val_set_size':10000,
        'encoding':'2d'
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
        'channels':13,
        'kernel':2,
        'batchnorm':True,
        'conv_layers':1,
    }
    training_params = {
        'episodes':5
    }
    agent_params['network_params'] = network_params
    save_data = False
    if save_data == True:
        save_data(params)

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

    results = compare_networks(agent_param_list,training_params)
    graph_networks(results)