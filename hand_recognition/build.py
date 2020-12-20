import numpy as np
import torch
import os
from random import shuffle
from itertools import combinations

import datatypes as dt
from data_utils import save_data,save_all
from cardlib import encode,decode,winner,hand_rank,rank
from card_utils import to_2d,suits_to_str,convert_numpy_to_rust,convert_numpy_to_2d,to_52_vector,swap_suits,build_52_key,save_obj
from create_hands import straight_flushes,quads,full_houses,flushes,straights,trips,two_pairs,one_pairs,high_cards,hero_5_cards,sort_hand

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
        Builds regression datasets with a target of win,loss,tie [1,-1,0]
        """
        if params['datatype'] == dt.DataTypes.THIRTEENCARD:
            trainX,trainY = self.build_13card(params[dt.Globals.INPUT_SET_DICT['train']],params['encoding'])
            valX,valY = self.build_13card(params[dt.Globals.INPUT_SET_DICT['val']],params['encoding'])
        elif params['datatype'] == dt.DataTypes.TENCARD:
            trainX,trainY = self.build_10card(params[dt.Globals.INPUT_SET_DICT['train']],params['encoding'])
            valX,valY = self.build_10card(params[dt.Globals.INPUT_SET_DICT['val']],params['encoding'])
        elif params['datatype'] == dt.DataTypes.PARTIAL:
            trainX,trainY = self.build_partial(params[dt.Globals.INPUT_SET_DICT['train']])
            valX,valY = self.build_partial(params[dt.Globals.INPUT_SET_DICT['val']])
        else:
            raise ValueError(f"Datatype {params['datatype']} not understood")
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
        Builds categorical targets of hand class.

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
        for dataset in ['train','val']:
            save_path = os.path.join(params['save_dir'],dataset)
            xpath = f"{os.path.join(save_path,dataset)}X"
            ypath = f"{os.path.join(save_path,dataset)}Y"
            X = []
            y = []
            num_hands = params[dt.Globals.INPUT_SET_DICT[dataset]] // 9
            if params['datatype'] == dt.DataTypes.NINECARD:
                for category in dt.Globals.HAND_TYPE_DICT.keys():
                    print('category',category)
                    for _ in range(num_hands):
                        hand,board = self.create_ninecard_handtypes(category)
                        shuffled_hand,shuffled_board = CardDataset.shuffle_hand_board(hand,board)
                        x_input = np.concatenate([shuffled_hand,shuffled_board],axis=0)
                        X.append(x_input)
                        y.append(category)
            elif params['datatype'] == dt.DataTypes.FIVECARD:
                for category in dt.Globals.HAND_TYPE_DICT.keys():
                    print('category',category)
                    for _ in range(num_hands):
                        X.append(self.create_handtypes(category))
                        y.append(category)
            else:
                raise ValueError(f"{params['datatype']} datatype not understood")
            X = np.stack(X)
            y = np.stack(y)
            save_data(X,xpath)
            save_data(y,ypath)

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
        assert(False not in [(s[1]  > dt.SUITS.LOW-1 and s[1] < dt.SUITS.HIGH) == True for s in hand]),f'hand outside range {hand}'
        assert(False not in [(s[1]  > dt.SUITS.LOW-1 and s[1] < dt.SUITS.HIGH) == True for s in board]),f'board outside range {board}'
        en_hand = [encode(c) for c in hand]
        en_board = [encode(c) for c in board]
        hand_strength = hand_rank(en_hand,en_board)
        hand_type = CardDataset.find_strength(hand_strength)
        while hand_type != category:
            extra_cards_52 = np.random.choice(remaining_deck,4,replace=False)
            extra_cards_2d = to_2d(extra_cards_52)
            hand = np.concatenate([initial_cards[:2],extra_cards_2d[:2]],axis=0)
            board = np.concatenate([initial_cards[2:],extra_cards_2d[2:]],axis=0)
            assert(False not in [(s[1]  > dt.SUITS.LOW-1 and s[1] < dt.SUITS.HIGH) == True for s in hand]),f'hand outside range {hand}'
            assert(False not in [(s[1]  > dt.SUITS.LOW-1 and s[1] < dt.SUITS.HIGH) == True for s in board]),f'board outside range {board}'
            en_hand = [encode(c) for c in hand]
            en_board = [encode(c) for c in board]
            hand_strength = hand_rank(en_hand,en_board)
            hand_type = CardDataset.find_strength(hand_strength)
        return hand,board,hand_strength

    def create_thirteen_handtypes(self,category):
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
        assert(False not in [(s[1]  > dt.SUITS.LOW-1 and s[1] < dt.SUITS.HIGH) == True for s in hand]),f'hand outside range {hand}'
        assert(False not in [(s[1]  > dt.SUITS.LOW-1 and s[1] < dt.SUITS.HIGH) == True for s in board]),f'board outside range {board}'
        en_hand = [encode(c) for c in hand]
        en_board = [encode(c) for c in board]
        hand_strength = hand_rank(en_hand,en_board)
        hand_type = CardDataset.find_strength(hand_strength)
        while hand_type != category:
            extra_cards_52 = np.random.choice(remaining_deck,4,replace=False)
            extra_cards_2d = to_2d(extra_cards_52)
            hand = np.concatenate([initial_cards[:2],extra_cards_2d[:2]],axis=0)
            board = np.concatenate([initial_cards[2:],extra_cards_2d[2:]],axis=0)
            assert(False not in [(s[1]  > dt.SUITS.LOW-1 and s[1] < dt.SUITS.HIGH) == True for s in hand]),f'hand outside range {hand}'
            assert(False not in [(s[1]  > dt.SUITS.LOW-1 and s[1] < dt.SUITS.HIGH) == True for s in board]),f'board outside range {board}'
            en_hand = [encode(c) for c in hand]
            en_board = [encode(c) for c in board]
            hand_strength = hand_rank(en_hand,en_board)
            hand_type = CardDataset.find_strength(hand_strength)
        print(remaining_deck,type(remaining_deck))
        print(extra_cards_52,type(extra_cards_52))
        print(remaining_deck - extra_cards_52)
        return hand,board,hand_strength

    def build_thirteencard_handtypes(self,multiplier):
        """
        Hero hand, villain hand and board in various configurations. 
        E.G. board from all street stages. Designed for training hand distribution nets.
        """
        switcher = {
            0: straight_flushes,
            1: quads,
            2: full_houses,
            3: flushes,
            4: straights,
            5: trips,
            6: two_pairs,
            7: one_pairs,
            8: high_cards
        }
        X = []
        y = []
        for category in range(0,9):
            five_hands = switcher[category]()
            for hand in five_hands:
                deck = np.arange(1,53)
                # convert five to 52
                flat_hand = (hand[0]-1)+(13*(hand[1]-1))
                remainder_deck = np.array(list(set(deck) - set(flat_hand)))
                hero_hands = hero_5_cards(hand)
                for h in hero_hands:
                    combined,handtype,hero_rank,vil_rank = CardDataset.sample_hand_board(remainder_deck,np.transpose(h))
                    while handtype != category:
                        combined,handtype,hero_rank,vil_rank = CardDataset.sample_hand_board(remainder_deck,np.transpose(h))
                    # print(combined)
                    X.append(combined)
                    y.append((hero_rank,vil_rank))
                break
            break
        X = np.stack(X)
        y = np.stack(y)
        mask = np.random.shuffle(np.arange(len(y)))
        return X[mask,:].reshape(X.shape),y[mask].reshape(y.shape)

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
            result = 1 if ace_suit[0] == list(board_suit) else 0
            X_input = np.concatenate([shuffled_hand,shuffled_board],axis=0)
            X.append(X_input)
            y.append(result)
        X = np.stack(X)
        y = np.stack(y)[:,None]
        return X,y

    def build_partial(self,iterations):
        """
        inputs consistenting of hand + board during all streets
        4+padding all the way to the full 9 cards. Always evaluated vs random hand.
        inputs are sorted for data effeciency
        target = {-1,0,1}
        """
        X = []
        y = []
        for category in dt.Globals.HAND_TYPE_DICT.keys():
            for _ in range(iterations // 9):
                hero_hand,board,_ = self.create_ninecard_handtypes(category)
                ninecards = np.concatenate([hero_hand,board],axis=0)
                flat_card_vector = to_52_vector(ninecards)
                available_cards = list(set(self.deck) - set(flat_card_vector))
                flat_vil_hand = np.random.choice(available_cards,4,replace=False)
                vil_hand = np.array(to_2d(flat_vil_hand))
                en_hand = [encode(c) for c in hero_hand]
                en_vil = [encode(c) for c in vil_hand]
                en_board = [encode(c) for c in board]
                result = winner(en_hand,en_vil,en_board)
                # hand + board at all stages. Shuffle cards so its more difficult for network
                np.random.shuffle(hero_hand)
                pure_hand = np.concatenate([hero_hand,np.zeros((5,2))],axis=0)
                np.random.shuffle(hero_hand)
                hand_flop = np.concatenate([hero_hand,board[:3],np.zeros((2,2))],axis=0)
                np.random.shuffle(hero_hand)
                hand_turn = np.concatenate([hero_hand,board[:4],np.zeros((1,2))],axis=0)
                X.append(pure_hand)
                X.append(hand_flop)
                X.append(hand_turn)
                X.append(ninecards)
                y.append(result)
                y.append(result)
                y.append(result)
                y.append(result)
        X = np.stack(X)
        y = np.stack(y)[:,None]
        return X,y

    def build_flush(self):
        switcher = {
            0: straight_flushes,
            1: quads,
            2: full_houses,
            3: flushes,
            4: straights,
            5: trips,
            6: two_pairs,
            7: one_pairs,
            8: high_cards
        }
        X = []
        y = []
        for category in [0,4]:
            five_hands = switcher[category]()
            for hand in five_hands:
                # Run through all padded versions
                hero_hands = hero_5_cards(hand)
                for h in hero_hands:
                    en_hand = [encode(c) for c in h]
                    flat_hand = np.transpose(sort_hand(np.transpose(h)))
                    compressed = to_52_vector(flat_hand) + 1
                    X.append(compressed)
                    y.append(rank(en_hand))
        X = np.stack(X)
        y = np.stack(y)
        mask = np.random.shuffle(np.arange(len(y)))
        return X[mask,:].reshape(X.shape),y[mask].reshape(y.shape)

    def build_smalldeck(self):
        """Complete dataset for preflop all the way to river for classifying all 5 card hands."""
        smalldeck = []
        for r in range(2,10):
            for s in range(1,3):
                smalldeck.append([r,s])
        hero_hands = np.array(list(combinations(smalldeck,4)))
        M = hero_hands.shape[0]
        zero_padding = np.zeros((M,5,2))
        X = np.concatenate([hero_hands,zero_padding],axis=1)
        X = X.reshape(M,-1)
        y = np.arange(M)
        return X,y

    def build_flatdeck(self,val=False):
        """Complete dataset for preflop all the way to river for classifying all 5 card hands."""
        switcher = {
            0: straight_flushes,
            1: quads,
            2: full_houses,
            3: flushes,
            4: straights,
            5: trips,
            6: two_pairs,
            7: one_pairs,
            8: high_cards
        }
        X = []
        y = []
        for category in range(0,9):
            five_hands = switcher[category]()
            for hand in five_hands:
                # Run through all padded versions
                if val:
                    hand = np.transpose(hand)
                    en_hand = [encode(c) for c in hand]
                    flat_hand = np.transpose(sort_hand(np.transpose(hand)))
                    compressed = to_52_vector(flat_hand) + 1
                    X.append(compressed)
                    y.append(rank(en_hand))
                else:
                    hero_hands = hero_5_cards(hand)
                    for h in hero_hands:
                        en_hand = [encode(c) for c in h]
                        flat_hand = np.transpose(sort_hand(np.transpose(h)))
                        compressed = to_52_vector(flat_hand) + 1
                        X.append(compressed)
                        y.append(rank(en_hand))
        X = np.stack(X)
        y = np.stack(y)
        mask = np.random.shuffle(np.arange(len(y)))
        return X[mask,:].reshape(X.shape),y[mask].reshape(y.shape)
        
    def build_hand_ranks_five(self,reduce_suits=True,valset=False):
        """
        rank 5 card hands
        input 5 cards
        target = {0-7462}
        """
        switcher = {
            0: straight_flushes,
            1: quads,
            2: full_houses,
            3: flushes,
            4: straights,
            5: trips,
            6: two_pairs,
            7: one_pairs,
            8: high_cards
        }
        X = []
        y = []
        for category in dt.Globals.HAND_TYPE_DICT.keys():
            if valset:
                five_hands = switcher[category]()
                for hand in five_hands:
                    sorted_hand = np.transpose(sort_hand(hand))
                    en_hand = [encode(c) for c in sorted_hand]
                    X.append(sorted_hand)
                    y.append(rank(en_hand))
            else:
                five_hands = switcher[category]()
                for hand in five_hands:
                    hero_hands = hero_5_cards(hand)
                    for h in hero_hands:
                        en_hand = [encode(c) for c in h]
                        X.append(np.transpose(sort_hand(np.transpose(h))))
                        y.append(rank(en_hand))
            # hero = hand[:2]
            # board = hand[2:]
            # hero = hero[np.argsort(hero[:,1]),:]
            # board = board[np.argsort(board[:,1]),:]
            # hero = hero[np.argsort(hero[:,0]),:]
            # board = board[np.argsort(board[:,0]),:]
            # hand = np.concatenate([hero,board])
            # if reduce_suits:
            #     hand = swap_suits(hand)
            # hand = hand[np.argsort(hand[:,0]),:] # lost blocker info
        X = np.stack(X)
        y = np.stack(y)
        return X,y

    def build_hand_ranks_nine(self,multiplier):
        """
        rank 5 card hands
        input 5 cards
        target = {0-7462}
        """
        Number_of_examples = {
            0:10,
            1:156,
            2:156,
            3:1277,
            4:10,
            5:858,
            6:858,
            7:2860,
            8:1277
        }
        X = []
        y = []
        for category in dt.Globals.HAND_TYPE_DICT.keys():
            for _ in range(Number_of_examples[category] * multiplier):
                hand,board,hand_strength = self.create_ninecard_handtypes(category)
                for i in range(4):
                    np.random.shuffle(board)
                    if i == 0:
                        new_board = np.zeros((5,2))
                    elif i == 1:
                        new_board = np.concatenate([board[:3,:],np.zeros((2,2))],axis=0)
                    elif i == 2:
                        new_board = np.concatenate([board[:4,:],np.zeros((1,2))],axis=0)
                    else:
                        new_board = board
                    X.append(np.concatenate([hand,new_board],axis=0))
                    y.append(hand_strength)
        X = np.stack(X)
        y = np.stack(y)
        return X,y
        
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
        suits = np.arange(dt.SUITS.LOW,dt.SUITS.HIGH)
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
    def sample_hand_board(remainder_deck,fivecards):
        remainder_cards = np.random.choice(remainder_deck,8,replace=False)
        test_cards = np.transpose(to_2d(remainder_cards-1))
        hand = np.concatenate([fivecards[:,:2],test_cards[:,:2]],axis=-1)
        board = np.concatenate([fivecards[:,2:],test_cards[:,2:4]],axis=-1)
        vil_hand = test_cards[:,4:]
        en_vil = [encode(c) for c in np.transpose(vil_hand)]
        en_hand = [encode(c) for c in np.transpose(hand)]
        en_board = [encode(c) for c in np.transpose(board)]
        hand_strength = hand_rank(en_hand,en_board)
        vil_strength = hand_rank(en_vil,en_board)
        hand_type = CardDataset.find_strength(hand_strength)
        return np.concatenate([hand,vil_hand,board],axis=-1),hand_type,hand_strength,vil_strength


    @staticmethod
    def to_torch(inputs:list):
        return [torch.Tensor(x) for x in inputs]
