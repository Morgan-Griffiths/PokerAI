import numpy as np
import torch
from collections import deque
import os
from random import shuffle

import hand_recognition.datatypes as dt
from cardlib import encode,decode,winner,hand_rank,rank
from card_utils import to_2d,suits_to_str,convert_numpy_to_rust,convert_numpy_to_2d,to_52_vector

class CardDataset(object):
    def __init__(self,params):
        self.deck = np.arange(52)
        self.suit_types = np.arange(dt.SUITS.LOW,dt.SUITS.HIGH)
        self.rank_types = np.arange(dt.RANKS.LOW,dt.RANKS.HIGH)
        self.alphabet_suits = ['s','h','d','c']
        self.suit_dict = {suit:alpha for suit,alpha in zip(self.suit_types,self.alphabet_suits)}
        self.fivecard_indicies = np.arange(5)
        self.params = params

    def generate_dataset(self,params):
        """
        Hands in test set may or may not match hands in training set.
        """
        if params['datatype'] == dt.DataTypes.THIRTEENCARD:
            trainX,trainY = self.build_13card(params[dt.Globals.INPUT_SET_DICT['train']],params['encoding'])
            valX,valY = self.build_13card(params[dt.Globals.INPUT_SET_DICT['test']],params['encoding'])
        if params['datatype'] == dt.DataTypes.TENCARD:
            trainX,trainY = self.build_10card(params[dt.Globals.INPUT_SET_DICT['train']],params['encoding'])
            valX,valY = self.build_10card(params[dt.Globals.INPUT_SET_DICT['test']],params['encoding'])
        trainX,trainY,valX,valY = CardDataset.to_torch([trainX,trainY,valX,valY])
        print(f'trainX: {trainX.shape}, trainY {trainY.shape}, valX {valX.shape}, valY {valY.shape}')
        return trainX,trainY,valX,valY

    def build_13card(self,iterations,encoding):
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

    def build_10card(self,iterations,encoding):
        """
        Generates X = (i,10,2) y = [-1,0,1]
        """
        X,y = [],[]
        for i in range(iterations):
            category = np.random.choice(np.arange(9))
            hand1 = self.create_handtypes(category)
            hand2 = self.create_handtypes(category)
            encoded_hand1 = [encode(c) for c in hand1]
            encoded_hand2 = [encode(c) for c in hand2]
            hand1_rank = rank(encoded_hand1)
            hand2_rank = rank(encoded_hand2)
            if hand1_rank > hand2_rank:
                result = 1
            elif hand1_rank < hand2_rank:
                result = -1
            else:
                result = 0
            X.append(np.vstack((hand1,hand2)))
            y.append(result)
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
        for dataset in ['train','test']:
            save_path = os.path.join(params['save_dir'],dataset)
            num_hands = params[dt.Globals.INPUT_SET_DICT[dataset]] // 9
            hand_strengths = {i:deque(maxlen=num_hands) for i in range(0,9)}
            if params['datatype'] == dt.DataTypes.NINECARD:
                for category in dt.Globals.HAND_TYPE_DICT.keys():
                    print('category',category)
                    for _ in range(num_hands):
                        hand,board = self.create_ninecard_handtypes(category)
                        shuffled_hand,shuffled_board = CardDataset.shuffle_hand_board(hand,board)
                        x_input = np.concatenate([shuffled_hand,shuffled_board],axis=0)
                        hand_strengths[category].append(x_input)
            elif params['datatype'] == dt.DataTypes.FIVECARD:
                for category in dt.Globals.HAND_TYPE_DICT.keys():
                    for _ in range(num_hands):
                        hand_strengths[category].append(self.create_handtypes(category))
            else:
                raise ValueError(f"{params['datatype']} datatype not understood")
            [print(len(hand_strengths[i])) for i in range(0,9)]
            for i in range(0,9):
                np.save(os.path.join(save_path,f'Hand_type_{dt.Globals.HAND_TYPE_DICT[i]}'),hand_strengths[i])

    def create_ninecard_handtypes(self,category):
        """
        Grab 5 cards representing that handtype, then split into hand/board and add cards, 
        if handtype hasn't changed store hand.
        """
        initial_cards = self.create_handtypes(category)
        flat_card_vector = to_52_vector(initial_cards)
        remaining_deck = list(set(self.deck) - set(flat_card_vector))
        extra_cards_52 = np.random.choice(remaining_deck,4,replace=False)
        extra_cards_2d = to_2d(extra_cards_52)
        hand = np.concatenate([initial_cards[:2],extra_cards_2d[:2]],axis=0)
        board = np.concatenate([initial_cards[2:],extra_cards_2d[2:]],axis=0)
        en_hand = [encode(c) for c in hand]
        en_board = [encode(c) for c in board]
        hand_strength = hand_rank(en_hand,en_board)
        hand_type = CardDataset.find_strength(hand_strength)
        while hand_type != category:
            extra_cards_52 = np.random.choice(remaining_deck,4,replace=False)
            extra_cards_2d = to_2d(extra_cards_52)
            hand = np.concatenate([initial_cards[:2],extra_cards_2d[:2]],axis=0)
            board = np.concatenate([initial_cards[2:],extra_cards_2d[2:]],axis=0)
            en_hand = [encode(c) for c in hand]
            en_board = [encode(c) for c in board]
            hand_strength = hand_rank(en_hand,en_board)
            hand_type = CardDataset.find_strength(hand_strength)
        return hand,board

    def build_blockers(self,iterations):
        """
        board always flush. Hand either has A blocker or A no blocker. Never has flush
        """
        X = []
        y = []
        for _ in range(iterations):
            ranks = np.arange(2,14)
            board = np.random.choice(ranks,5,replace=False)
            board_suits = np.full(5,np.random.choice(self.suit_types))
            hand = np.random.choice(ranks,3,replace=False)
            board_suit = set(board_suits)
            other_suits = set(self.suit_types).difference(board_suit)
            hand_suits = np.random.choice(list(other_suits),3)
            ace = np.array([14])
            ace_suit_choices = [list(board_suit)[0],list(other_suits)[0]]
            ace_suit = np.random.choice(ace_suit_choices,1)
            hand_ranks = np.hstack((ace,hand))
            hand_suits = np.hstack((ace_suit,hand_suits))
            hand = np.stack((hand_ranks,hand_suits),axis=-1)
            board = np.stack((board,board_suits),axis=-1)
            shuffled_hand,shuffled_board = CardDataset.shuffle_hand_board(hand,board)
            result = 1 if ace_suit == board_suit else 0
            X_input = np.concatenate([shuffled_hand,shuffled_board],axis=0)
            X.append(X_input)
            y.append(result)
        X = np.stack(X)
        y = np.stack(y)[:,None]
        return X,y

    def build_partial(self):
        """
        inputs consistenting of hand + board during all streets
        target = {-1,0,1}
        """
        pass

    def build_hand_ranks(self):
        """
        rank 5 card hands
        input 5 cards
        target = {0-7462}
        """
        pass
        # for category in dt.Globals.HAND_TYPE_DICT.keys():
        #     for _ in range(num_hands):
        #         hand_strengths[category].append(self.create_handtypes(category))
        
    def create_handtypes(self,category,randomize=True):
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
        if randomize == True:
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
        # Avoid straight flushes. Wheel is special case
        if set(ranks4) == set([14,2,3,4]) or set(ranks4) == set([2,3,4,5]) or set(ranks4) == set([14,3,4,5]) or set(ranks4) == set([14,2,4,5]) or set(ranks4) == set([14,2,3,5]):
            wheel_cards = np.array([14,2,3,4,5,6])
            untouchables = set(wheel_cards) & set(self.rank_types)
            possible = set(self.rank_types) - set(untouchables)
        elif np.max(ranks4) - np.min(ranks4) < 6:
            untouchables = set(ranks4) | set(np.arange(np.min(ranks4)-1,np.max(ranks4)+2)) & set(self.rank_types)
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
        first_pair_suits = np.random.choice(self.suit_types,2,replace=False)
        second_pair_suits = np.random.choice(self.suit_types,2,replace=False)
        last_suit  = np.random.choice(self.suit_types,1)
        suits = np.hstack((first_pair_suits,second_pair_suits,last_suit))
        ranks = np.hstack((first_pair,second_pair,other_rank))
        hand = np.stack((ranks,suits))
        return hand

    def one_pair(self):
        first_pair = np.full(2,np.random.randint(dt.RANKS.LOW,dt.RANKS.HIGH))
        other_ranks = np.random.choice(list(set(self.rank_types).difference(set(first_pair))),3,replace=False)
        ranks = np.hstack((first_pair,other_ranks))
        suits2 = np.random.choice(self.suit_types,2,replace=False)
        rest_suits = np.random.choice(self.suit_types,3)
        suits = np.hstack((suits2,rest_suits))
        hand = np.stack((ranks,suits))
        return hand

    def high_card(self):
        ranks4 = np.random.choice(self.rank_types,4,replace=False)
        if set(ranks4) == set([14,2,3,4]) or set(ranks4) == set([2,3,4,5]) or set(ranks4) == set([14,3,4,5]) or set(ranks4) == set([14,2,4,5]) or set(ranks4) == set([14,2,3,5]):
            wheel_cards = np.array([14,2,3,4,5,6])
            untouchables = set(wheel_cards) & set(self.rank_types)
            possible = set(self.rank_types) - set(untouchables)
        elif np.max(ranks4) - np.min(ranks4) < 6:
            untouchables = set(ranks4) | set(np.arange(np.min(ranks4)-1,np.max(ranks4)+2)) & set(self.rank_types)
            possible = set(self.rank_types) - set(untouchables)
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
    def shuffle_hand_board(hand,board):
        hand_indicies = np.arange(4)
        board_indicies = np.arange(5)
        np.random.shuffle(hand_indicies)
        np.random.shuffle(board_indicies)
        shuffled_board = board[board_indicies,:]
        shuffled_hand = hand[hand_indicies,:]
        return shuffled_hand,shuffled_board

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
