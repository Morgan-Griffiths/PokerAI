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

def showdown(handA,handB):
    if handA > handB:
        return handA
    else:
        return handB

class GameRules(object):
    def __init__(self,ante,players,deck):
        self.ante = ante
        self.players = players
        self.deck = deck

class GameTree(object):
    def __init__(self,info_set,rules,root):
        self.info_set = {}
        self.rules = deepcopy(rules)
        self.root = None

    def build(self):
        players_in = True * self.rules.players
        pot = [self.rules.ante] * self.rules.players
        bets = [0] * self.rules.players
        bet_history = ""
        self.root = build_rounds(self, )

    def build_rounds(self):

    def deal_holecards(self):
    deal cards - build those nodes, update info_set

    def add_node(self):

class Node(object):
    def __init__(self,parent,holecards,bet_history):
        self.holecards = deepcopy(holecards)
        self.children = children

        if parent:
            self.parent = parent
            self.parent.add_child(self)

    def add_child(self,child):
        if self.children = None:
            self.children = [child]
        else:
            children.append(child)


class ChanceNode(node):
    def __init__(self,parent,holecards,bet_history):
        Node.__init__(self,parent,holecards,bet_history):

class HolecardNode(node):
    def __init__(self,parent,holecards,bet_history):
        Node.__init__(self,parent,holecards,bet_history):

class ActionNode(node):
    def __init__(self,parent,holecards,bet_history):
        Node.__init__(self,parent,holecards,bet_history):

class TerminalNode(node)
    def __init__(self,parent,holecards,bet_history):
        Node.__init__(self,parent,holecards,bet_history):

        #account for showdown and win/loss
