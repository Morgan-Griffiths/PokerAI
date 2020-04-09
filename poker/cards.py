
from cardlib import encode,decode,winner
from random import shuffle
import numpy as np
from torch import optim
from collections import deque
import torch
import torch.nn.functional as F
import timeit
from models.networks import CardClassification
from agents.agent import CardAgent
import os

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

def train_network(data,agent,params):
    train_scores = deque(maxlen=50)
    val_scores = deque(maxlen=50)
    for i in range(1,101):
        preds = agent(data['trainX'])
        loss = agent.backward(preds,data['trainY'])
        val_preds = agent.predict(data['valX'])
        val_loss = F.smooth_l1_loss(val_preds,data['valY']).item()
        train_scores.append(loss)
        val_scores.append(val_loss)
        if i % 10 == 0:
            print('loss',np.mean(train_scores))
            print('val loss',np.mean(val_scores))
    agent.save_weights(os.path.join(params['save_path'],params['agent_name']))

if __name__ == "__main__":
    params = {
        'training_set_size':50000,
        'val_set_size':10000,
        'encoding':'2d'
    }
    agent_params = {
        'learning_rate':2e-3
    }
    training_params = {
        'episodes':100,
        'save_path':'checkpoints',
        'agent_name':'Conv(kernel2)'
    }
    save_data = False
    seed = 346
    state_space = (13,2)
    if save_data == True:
        dataset = CardDataset(params)
        trainX,trainY,valX,valY = dataset.generate_dataset(params)
        np.save('data/trainX',trainX)
        np.save('data/trainY',trainY)
        np.save('data/valX',valX)
        np.save('data/valY',valY)

    trainX = np.load('data/trainX.npy')
    trainY = np.load('data/trainY.npy')
    valX = np.load('data/valX.npy')
    valY = np.load('data/valY.npy')
    
    print(f'trainX: {trainX.shape}, trainY {trainY.shape}, valX {valX.shape}, valY {valY.shape}')
    #to torch
    trainX,trainY,valX,valY = CardDataset.to_torch([trainX,trainY,valX,valY])

    data = {
        'trainX':trainX,
        'trainY':trainY,
        'valX':valX,
        'valY':valY,
    }

    agent = CardAgent(state_space,seed,agent_params)
    train_network(data,agent,training_params)
