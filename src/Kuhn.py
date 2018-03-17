import math
import numpy as np
import random
import itertools as it
from copy import deepcopy

"""
Initalize game with kuhn_deck and all combinations of holecards dealt. Antes collected into pot.
Each holecard has a given probability 1/3. with 6 possible combinations of hands. 12 info sets.
6 game states. up to 3 rounds. 6 holecard nodes. from each holecard node two action child nodes.

"""

kuhn_values = {"A" : 14, "K" : 13, "Q" : 12}
kuhn_deck = ("A","K","Q")

BET = 1
CHECK = 0
actions = ['check','bet']
A = [0,1]
NUM_ACTIONS = global 2

class Player(object):
    def __init__(self, choice, action_vector, regret_vector):
        self.choice = choice
        self.action_vector = action_vector
        self.regret_vector = regret_vector

class GameTree(object):
    def __init__(self,deck,max_bets):
        self.deck = deck
        self.max_bets = max_bets
        self.actions = [0,1]

        def get_node(info_set):



class Node(object):
    def __init__(self,info_set):
        self.info_set = None
        self.regret = np.zeros(2)
        self.strategy = np.zeros(2)
        self.strategySum = np.zeros(2)

        def regret(self,action):


        def get_strategy(self,weight):
            normalizing_sum = 0
            for i in xrange(0,NUM_ACTIONS):
                if regretSum[i] > 0:
                    strategy[i] = regretSum[i]
                else:
                    strategy[i] = 0
                normalizing_sum += strategy[i]
            for x in xrange(0,NUM_ACTIONS):
                if normalizing_sum > 0:
                    strategy[x] = strategy[x]/normalizing_sum
                else:
                    strategy[x] = 1 / NUM_ACTIONS

        def get_AvgStrategy(self):

        def get_infoset(self,info_set):
            if info_set == None:
                node = Node(info_set)
                nodeMap.append(Node)

class ChanceNode(Node):
    def __init__(self,info_set):
        Node.__init__(self,info_set)

class TerminalNode(Node):
    def __init__(self,info_set):
        Node.__init__(self,info_set)

        def payoff(hand1,hand2,info_set):



def cfr(h,i,t,policy_1,policy_2):
    if h == terminal:
        return payoff(h)
    elif h == chanceNode:
        sample outcome a ~ sigma_chance (h,a)
        return cfr(h,i,t,policy_1,policy_2)
    I = info_set(h)
    v = 0


def showdown(handA,handB):
    if handA > handB:
        return handA
    else:
        return handB

def

def utility(handA,handB,deck):

regret_table = np.zeros
strategy_table = np.zeros((2,2))
profile = 0
