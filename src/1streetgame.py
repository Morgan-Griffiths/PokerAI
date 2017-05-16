import Classes as cl
import Pokereval as pe
import random
import math
import copy
import itertools
import collections
import functools

def default_infoset_format(player, holecards, bet_history):
    return "({0}{1}:{2}):".format("".join([str(x) for x in holecards]),"".join([str(x) for x in board]),bet_history)

class GameRules(object):
    def __init__(self, players, deck, rounds, ante, blinds, \
    handeval = pe.main,infoset_format = default_infoset_format):
    assert(players >= 2)
    assert(ante >=0)
    assert(rounds != None)
    assert(deck != None)
    assert(len(rounds) > 0)
    assert(len(deck) > 1)
    for r in rounds:
        assert(len(r.maxbets) == players)
    self.players = players
    self.deck = deck
    self.rounds = rounds
    self.ante = ante
    self.blinds = blinds
    self.handeval = handeval
    self.infoset_format = infoset_format

class RoundInfo(object):
    def __init__(self, holecards, boardcards, betsize, maxbets):
        self.holecards = holecards
        self.boardcards = boardcards
        self.betsize = betsize
        self.maxbets = maxbets
