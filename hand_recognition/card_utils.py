import numpy as np
import copy
import pickle
from datatypes import SUITS,RANKS

SUIT_DICT = {
    1:'s',
    2:'h',
    3:'d',
    4:'c'
}
REVERSE_SUIT_DICT = {v:k for k,v in SUIT_DICT.items()}
def save_obj(path,obj):
    with open(f'{path}.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(path):
    with open(f'{path}.pkl', 'rb') as f:
        return pickle.load(f)

def build_52_key(digits):
    """takes a series of 52 digit cards"""
    key = 0
    for i,digit in enumerate(digits,0):
        key += digit*(52**i)
    return key

def convert_flat_to_52(cards):
    new_cards = []
    for i in range(0,len(cards)-1,2):
        new_cards.append(cards[i]*cards[i+1])
    return new_cards

def swap_suits(cards):
  """
  Swap suits to remove most symmetries.

  Modifies cards in place

  Fails to remove some in the case where there is a lower-ranked
  pair or triple that shares a suit with a higher-ranked card.
  TODO: handle [2,3], [2,4], [6,3], [7,4] deterministically
  """
  cards_need_swap = cards
  new_suit = 5
  while cards_need_swap.shape[0] > 0:
    suit = cards_need_swap[0,1]
    cards[cards[:,1] == suit, 1] = new_suit
    new_suit += 1
    cards_need_swap = cards[cards[:,1] < 5]
  cards[:,1] = cards[:,1] - 4
  return cards

def convert_numpy_to_rust(vectors):
    cards = []
    for vector in vectors:
        np_suit = np.floor(np.divide(vector,13)).astype(int)
        rank = np.subtract(vector,np.multiply(np_suit,13))
        rank = np.add(rank,2)
        np_suit = np.add(np_suit,1)
        suit = SUIT_DICT[np_suit]
        cards.append([rank,suit])
    return cards

def convert_numpy_to_2d(vectors):
    cards = []
    for vector in vectors:
        np_suit = np.floor(np.divide(vector,13)).astype(int)
        rank = np.subtract(vector,np.multiply(np_suit,13))
        rank = np.add(rank,2)
        np_suit = np.add(np_suit,1)
        cards.append([rank,np_suit])
    return cards

def cards_to_planes(cards):
    new_cards = copy.deepcopy(cards)
    plane = np.ndarray((len(new_cards),2))
    for i,card in enumerate(new_cards):
        plane[i][0] = card[0]
        plane[i][1] = card[1]
    return plane
    
#2d
def suits_to_str(cards):
    new_cards = copy.deepcopy(cards)
    for card in new_cards:
        card[1] = SUIT_DICT[card[1]]
    return new_cards

#2d
def suits_to_num(cards):
    new_cards = copy.deepcopy(cards)
    for card in new_cards:
        card[1] = REVERSE_SUIT_DICT[card[1]]
    return new_cards

#takes 2d vector of numbers, turns into (1,4) matrix of numbers between 0-51
#returns np.array
def to_52_vector(vector):
    rank = np.transpose(vector)[:][0]
    suit = np.transpose(vector)[1][:]
    rank = np.subtract(rank,RANKS.LOW)
    suit = np.subtract(suit,SUITS.LOW)
    return np.add(rank,np.multiply(suit,13))

#takes (1,4) vector of numbers between 0-51 and turns into 2d vector of numbers between 0-13 and 1-4
#returns list
def to_2d(vector):
    if type(vector) == np.ndarray or type(vector) == list:
        suit = np.floor(np.divide(vector,13))
        suit = suit.astype(int)
        rank = np.subtract(vector,np.multiply(suit,13))
        rank = np.add(rank,RANKS.LOW)
        suit = np.add(suit,SUITS.LOW)
        combined = np.concatenate([rank,suit])
        length = int(len(combined) / 2)
        hand_length = len(vector)
        hand = [[combined[x],combined[x+hand_length]] for x in range(length)]
    else:
        suit = np.floor(np.divide(vector,13))
        suit = suit.astype(int)
        rank = np.subtract(vector,np.multiply(suit,13))
        rank = np.add(rank,RANKS.LOW)
        suit = np.add(suit,SUITS.LOW)
        hand = [[rank,suit]]
        print(hand,'hand')
    #print(hand,'combined')
    return hand
    
#takes (1,4) numpy vector of numbers between 0-51 and returns 1 hot encoded vector
#returns list of numpy vectors
def to_1hot(vect):
    hand = []
    for card in vect:
        vector = np.zeros(52)
        vector[card] = 1
        hand.append(vector)
    return hand

#takes (1,52) 1 hot encoded vector and makes it (1,53)
#returns np.array
def hot_pad(vector):
    temp = np.copy(vector)
    padding = np.reshape(np.zeros(len(temp)),(len(temp),1))
    temp = np.hstack((temp,padding))
    return temp

#Takes 1hot encoded vector
#returns (1,4) 52 encoded vector
def from_1hot(vect):
    new_hand = []
    for card in vect:
        #print(card)
        i, = np.where(card == 1)
        #print(i)
        new_hand.append(i)
    return new_hand

#Takes 1 hot encoded padded vector and returns 1 hot encoded vector
def remove_padding(vect):
    if len(vect[0]) != 53:
        raise ValueError("1 Hot vector must be padded")
    new_hand = []
    for card in vect:
        new_hand.append(card[:-1])
    return new_hand
    
def convert_str_to_1hotpad(hand):
    vector1 = suits_to_num(hand)
    vector2 = to_52_vector(vector1)
    vector3 = to_1hot(vector2)
    vector4 = hot_pad(vector3)
    return vector4

def convert_1hotpad_to_str(hand):
    vector_unpad = remove_padding(hand)
    vector_unhot = from_1hot(vector_unpad)
    vector_un1d = to_2d(vector_unhot)
    vector_unnum = suits_to_str(vector_un1d)
    return vector_unnum

def convert_52_to_str(hand):
    vector_un1d = to_2d(hand)
    vector_unnum = suits_to_str(vector_un1d)
    return vector_unnum