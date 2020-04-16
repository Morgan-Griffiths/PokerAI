import numpy as np
import torch
from collections import deque
import os

import hand_recognition.datatypes as dt
from cardlib import encode,decode,winner,hand_rank,rank
from card_utils import to_2d,suits_to_str,convert_numpy_to_rust,convert_numpy_to_2d

class CardDataset(object):
    def __init__(self,params):
        self.deck = np.arange(52)
        self.params = params

    def generate_dataset(self,params):
        """
        Hands in test set may or may not match hands in training set.
        """
        if params['datatype'] == dt.DataTypes.RANDOM:
            trainX,trainY = self.generate_hands(params['training_set_size'],params['encoding'])
            valX,valY = self.generate_hands(params['val_set_size'],params['encoding'])
        if params['datatype'] == dt.DataTypes.FIVECARD:
            trainX,trainY = self.build_5card(params['training_set_size'],params['encoding'])
            valX,valY = self.build_5card(params['val_set_size'],params['encoding'])
        trainX,trainY,valX,valY = CardDataset.to_torch([trainX,trainY,valX,valY])
        print(f'trainX: {trainX.shape}, trainY {trainY.shape}, valX {valX.shape}, valY {valY.shape}')
        return trainX,trainY,valX,valY

    def generate_hands(self,iterations,encoding):
        """
        Generates X = (i,13,2) y = [-1,0,1]
        """
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

    def build_5card(self,iterations,encoding):
        """
        Generates X = (i,5,2) y = [-1,0,1]
        """
        X,y = [],[]
        for i in range(iterations):
            cards = np.random.choice(self.deck,5,replace=False)
            rust_cards = convert_numpy_to_rust(cards)
            encoded_cards = [encode(c) for c in rust_cards]
            category = CardDataset.find_strength(rank(encoded_cards))
            if encoding == '2d':
                cards2d = convert_numpy_to_2d(cards)
                X.append(cards2d)
            else:
                X.append(cards)
            y.append(category)
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
        
        hand_strengths = {i:deque(maxlen=params['maxlen']) for i in range(0,9)}
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
        [print(len(hand_strengths[i])) for i in range(0,9)]
        for i in range(0,9):
            np.save(os.path.join(params['save_path'],f'Hand_type_{dt.DataTypes.HAND_TYPE_DICT[i]}'),hand_strengths[i])

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
