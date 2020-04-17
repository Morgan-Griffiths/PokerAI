import numpy as np
import torch
from collections import deque
import os
from random import shuffle

import hand_recognition.datatypes as dt
from cardlib import encode,decode,winner,hand_rank,rank
from card_utils import to_2d,suits_to_str,convert_numpy_to_rust,convert_numpy_to_2d

class CardDataset(object):
    def __init__(self,params):
        self.deck = np.arange(52)
        self.suit_types = np.arange(dt.SUITS.LOW,dt.SUITS.HIGH)
        self.rank_types = np.arange(dt.RANKS.LOW,dt.RANKS.HIGH)
        self.fivecard_indicies = np.arange(5)
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
        if params['datatype'] == dt.DataTypes.HANDTYPE:
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
                card_rank = hand_rank(en_hand1,en_board)
                category = CardDataset.find_strength(card_rank)
                hand_strengths[category].append(numpy_cards[:4]+numpy_cards[8:])
                hand_strengths[category]

                card_rank = hand_rank(en_hand2,en_board)
                category = CardDataset.find_strength(card_rank)
                hand_strengths[category].append(numpy_cards[4:8]+numpy_cards[8:])
        elif params['datatype'] == dt.DataTypes.FIVECARD:
            for category in dt.Globals.HAND_TYPE_DICT.keys():
                for _ in range(params['maxlen']):
                    hand_strengths[category].append(self.create_handtypes(category))
        else:
            raise ValueError(f"{params['datatype']} datatype not understood")
        [print(len(hand_strengths[i])) for i in range(0,9)]
        for i in range(0,9):
            np.save(os.path.join(params['save_path'],f'Hand_type_{dt.Globals.HAND_TYPE_DICT[i]}'),hand_strengths[i])

    def create_handtypes(self,category):
        switcher = {
            0: self.straight_flush,
            1: self.quads,
            2: self.full_house,
            3: self.flush,
            4: self.straight,
            5: self.trips,
            6: self.two_pair,
            7: self.one_pair,
            8: self.high_card
        }
        np.random.shuffle(self.fivecard_indicies)
        return np.transpose(switcher[category]()[:,self.fivecard_indicies])
         
    def straight_flush(self):
        # Pick a number from 13 - 5
        top = np.random.randint(dt.RANKS.LOW+5,dt.RANKS.HIGH)
        straight = np.arange(top-5,top)
        suit = np.random.choice(self.suit_types)
        hand = np.stack((straight,np.full(5,suit)))
        return hand

    def quads(self):
        quads = np.full(4,np.random.choice(self.rank_types))
        other_rank = np.random.choice(list(set(self.rank_types).difference(set(quads))))
        ranks = np.hstack((quads,other_rank))
        suits = np.arange(4)
        other_suit = np.random.choice(self.suit_types)
        suits = np.hstack((suits,other_suit))
        hand = np.stack((ranks,suits))
        return hand

    def full_house(self):
        trips = np.full(3,np.random.choice(self.rank_types))
        trip_suits = np.random.choice(self.suit_types,3,replace=False)
        pair = np.full(2,np.random.choice(list(set(self.rank_types).difference(set(trips)))))
        pair_suits = np.random.choice(self.suit_types,2,replace=False)
        ranks = np.hstack((trips,pair))
        suits = np.hstack((trip_suits,pair_suits))
        hand = np.stack((ranks,suits))
        return hand

    def flush(self):
        ranks4 = np.random.choice(self.rank_types,4,replace=False)
        if np.max(ranks4) - np.min(ranks4) == 4:
            untouchables = set(ranks4) & set((np.max(ranks4)+1,np.min(ranks4)-1)) & set(self.rank_types)
            possible = set(self.rank_types) - set(untouchables)
        else:
            possible = set(self.rank_types) - set(ranks4)
        last_rank = np.random.choice(list(possible))
        ranks = np.hstack((ranks4,last_rank))
        suits = np.full(5,np.random.choice(self.suit_types))
        hand = np.stack((ranks,suits))
        return hand

    def straight(self):
        top = np.random.randint(dt.RANKS.LOW+5,dt.RANKS.HIGH)
        straight = np.arange(top-5,top)
        suits = np.random.choice(self.suit_types,5,replace=True)
        if len(set(suits)) == 1:
            other_suit = np.random.choice(list(set(self.suit_types).difference(set(suits))))
            suits[np.random.choice(np.arange(5))] = other_suit
        hand = np.stack((straight,np.full(5,suits)))
        return hand

    def trips(self):
        trips = np.full(3,np.random.choice(self.rank_types))
        other_ranks = np.random.choice(list(set(self.rank_types).difference(set(trips))),2,replace=False)
        ranks = np.hstack((trips,other_ranks))
        trip_suits = np.random.choice(self.suit_types,3,replace=False)
        other_suits = np.random.choice(self.suit_types,2,replace=True)
        suits = np.hstack((trip_suits,other_suits))
        hand = np.stack((ranks,suits))
        return hand

    def two_pair(self):
        first_pair = np.full(2,np.random.randint(dt.RANKS.LOW,dt.RANKS.HIGH))
        second_pair = np.full(2,np.random.choice(list(set(self.rank_types).difference(set(first_pair)))))
        other_rank = np.random.choice(list(set(self.rank_types).difference(set(first_pair)|set(second_pair))))
        ranks = np.hstack((first_pair,second_pair,other_rank))
        two_suits = np.random.choice(self.suit_types,2,replace=False)
        rest_suits  = np.random.choice(self.suit_types,3,replace=True)
        suits = np.hstack((two_suits,rest_suits))

        ranks = np.hstack((first_pair,second_pair,other_rank))
        hand = np.stack((ranks,suits))
        return hand

    def one_pair(self):
        first_pair = np.full(2,np.random.randint(dt.RANKS.LOW,dt.RANKS.HIGH))
        other_ranks = np.random.choice(list(set(self.rank_types).difference(set(first_pair))),3,replace=False)
        ranks = np.hstack((first_pair,other_ranks))
        suits2 = np.random.choice(self.suit_types,2,replace=False)
        rest_suits = np.random.choice(self.suit_types,3,replace=False)
        suits = np.hstack((suits2,rest_suits))
        hand = np.stack((ranks,suits))
        return hand

    def high_card(self):
        ranks4 = np.random.choice(self.rank_types,4,replace=False)
        if np.max(ranks4) - np.min(ranks4) == 4:
            untouchables = set(ranks4) & set((max(ranks4)+1,min(ranks4)-1)) & set(self.rank_types)
            possible = set(self.rank_types) - untouchables
        else:
            possible = set(self.rank_types) - set(ranks4)
        last_rank = np.random.choice(list(possible))
        ranks = np.hstack((ranks4,last_rank))
        suits4 = np.random.choice(self.suit_types,4,replace=False)
        if len(set(suits4)) == 1:
            last_suit = np.random.choice(list(set(self.suit_types) - set(suits4)))
        else:
            last_suit = np.random.choice(self.suit_types)
        suits = np.hstack((suits4,last_suit))
        hand = np.stack((ranks,suits))
        return hand


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
