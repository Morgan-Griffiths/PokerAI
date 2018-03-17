from itertools import combinations
from itertools import permutations
from itertools import product
from collections import Counter
import numpy as np
from copy import deepcopy
from functools import partial
import Pokereval as pe

def default_infoset_format(player, holecards, board, bet_history):
    return "{0}{1}:{2}:".format("".join([str(x) for x in holecards]), "".join([str(x) for x in board]), bet_history)

class GameRules(object):
    def __init__(self,players,deck,rounds,ante,blinds,handeval = pe.main, infoset_format=default_infoset_format):
        if blinds != None:
            if type(blinds) is int or type(blinds) is float:
                blinds = [blinds]
        self.players = players
        self.deck = deck
        self.roundinfo = rounds
        self.ante = ante
        self.blinds = blinds
        self.infoset_format = infoset_format

class RoundInfo(object):
    def __init__(self, holecards, boardcards, betsize, maxbets):
        self.holecards = holecards
        self.boardcards = boardcards
        self.betsize = betsize
        self.maxbets = maxbets

class GameTree(object):
    def __init__(self,rules):
        self.rules = deepcopy(rules)
        self.infosets = {}
        self.root = None

    def build(self):
        # Assume everyone is in
        players_in = [True] * self.rules.players
        # Collect antes
        committed = [self.rules.ante] * self.rules.players
        bets = [0] * self.rules.players
        # Collect blinds
        next_player = self.collect_blinds(committed, bets, 0)
        holes = [] * self.rules.players
        board = []
        bet_history = ""
        self.root = self.build_rounds(None, players_in, committed, holes, board, self.rules.deck, bet_history, 0, bets, next_player)

    def collect_blinds(self, committed, bets, next_player):
        if self.rules.blinds != None:
            for blind in self.rules.blinds:
                committed[next_player] += blind
                bets[next_player] = int((committed[next_player] - self.rules.ante) / self.rules.roundinfo[0].betsize)
                next_player = (next_player + 1) % self.rules.players
        return next_player

    #def showdown(self, root, players_in, committed, holes, board, deck, bet_history):
    #    if players_in.count(True) == 1:


class Node(object):
    def __init__(self, parent, committed, holecards, board, deck, bet_history):
        self.committed = deepcopy(committed)
        self.holecards = deepcopy(holecards)
        self.board = deepcopy(board)
        self.deck = deepcopy(deck)
        self.bet_history = deepcopy(bet_history)
        if parent:
            self.parent = parent
            self.parent.add_child(self)

    def add_child(self,child):
        if self.children is None:
            self.children = [child]
        else:
            self.children.append(child)

class TerminalNode(Node):
    def __init__(self, parent, committed, holecards, board, deck, bet_history, payoffs, players_in):
        Node.__init__(self, parent, committed, holecards, board, deck, bet_history)
        self.payoff = payoff
        self.players_in = deepcopy(players_in)

class HolecardChanceNode(Node):
    def __init__(self, parent, committed, holecards, board, deck, bet_history, todeal):
        Node.__init__(self, parent, committed, holecards, board, deck, bet_history)
        self.todeal = todeal
        self.children = []

class ActionNode(Node):
    def __init__(self, parent, committed, holecards, board, deck, bet_history, player, infoset_format):
        Node.__init__(self, parent, committed, holecards, board, deck, bet_history)
        self.player = player
        self.children = []
        self.raise_action = None
        self.bet_action = None
        self.call_action = None
        self.fold_action = None

kuhn_values = {"A" : 14, "K" : 13, "Q" : 12}
kuhn_deck = ("A","K","Q")

"""
betsize can either be fixed, or a % of pot. if it is a % then it should be an array
thats populated by harmonic bet sizing. the frequency of the bet sizes should be
probabilistic.
"""

def overlap(t1, t2):
    for x in t1:
        if x in t2:
            return True
    return False

def all_unique(hc):
    for i in range(len(hc)-1):
        for j in range(i+1,len(hc)):
            if overlap(hc[i], hc[j]):
                return False
    return True

deck = kuhn_deck
players = 2
a = combinations(deck, 1)
print filter(lambda x: all_unique(x), permutations(a, players))

def main():
    players = 2
    deck = kuhn_deck
    rounds = [Roundinfo(holecards=1,boardcards=0,betsize=1,maxbets=[2,2]),\
    Roundinfo(holecards=0,boardcards=0,betsize=1,maxbets=[2,2])]
    ante = 1
    blinds = None
    gamerules = GameRules(players,deck,rounds,ante,blinds,handeval = pe.main)
    gametree = GameTree(gamerules)
    gametree.build()
