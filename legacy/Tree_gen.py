import numpy as np
import itertools
import re
import math
import pickle
import cardlib as cb
import Database_gen as dg
import Classes as cl
from Classes import Information_set
import os
import copy

"""
builds tree from series of infosets.
Steps through an infoset and creates nodes in each step
preorder, postorder, depth first

Create perspective field for Information_set?
Check for node before adding. If node exists add data to that node.

What are the keys?
infoset perspective
checking for nodes
creating limited complexity infosets
RL agent
"""
def openpickle():
    #to read file
    with open('infoset1.pickle', 'rb') as handle:
        b = pickle.load(handle)
    return b

def import_all(folder):
    pass
#TODO check for node before adding one
def build_tree(infoset):
    print infoset
    last_parent = ''
    #deck = copy.deepcopy(cl.DECK)
    #initialize root
    root = Node(None)
    #print "root"
    #find lowest stack = effective stacksize.
    eff_stack = min(infoset.positions_stacks, key = lambda x: x[1])[1]
    print eff_stack,infoset.positions_stacks
    if root.children != None:
        for x in xrange(0,len(root.children)):
            if root.children[x].stacksize == eff_stack:
                 match = root.children[x]
    #add stacksize children to root
    snode = StacksizeNode(root,eff_stack)
    if match:
        last_parent = match
    else:
        last_parent = snode
    print snode
    #print root.children
    #add holecard child to stacksize
    print infoset.hand_generic
    hnode = HolecardChanceNode(last_parent,infoset.hand_sorted,infoset.hand_generic)
    print hnode.holecards
    last_parent = hnode
    #print infoset.actions_full
    #two options. While there is a next option. or for len(action_full)
    #keep track of the previous node? so we can append
    #loop over action and board cards until reaching the terminal node
    for x in xrange(0,len(infoset.actions_full)):
        #if action, add action and betsize
        print "line",infoset.actions_full[x]
        #print type(infoset.actions_full[x])
        if type(infoset.actions_full[x]) == list:
            #board
            print 'board',infoset.actions_full[x]
            newboard = copy.deepcopy(infoset.actions_full[x])
            single = list(itertools.chain.from_iterable(newboard))
            #turn 1d list into concatenated str
            board_cards = ''.join(str(e) for e in single)
            print "board1", board_cards
            bnode = BoardcardChanceNode(last_parent,board_cards)
            last_parent = bnode
        else:
            #action
            print 'action1',infoset.actions_full[x]
            position = re.search(r"(.*)\s",infoset.actions_full[x])
            print 'position',position.group(1)
            action = re.search(r"\s([a-z])",infoset.actions_full[x])
            print 'action', action.group(1)
            anode = ActionNode(last_parent,action)
            last_parent = anode
            if not re.search(r"folds",infoset.actions_full[x]) and not re.search(r"checks",infoset.actions_full[x]):
                #add betsize
                amount = re.search(r"\s([0-9][.][0-9]{1,5})",infoset.actions_full[x])
                betsize = round(float(amount.group(1)),2)
                print "amount",amount.group(1),betsize
                betnode = BetsizeNode(last_parent,betsize)
                last_parent = betnode
    #add terminal node for outcome
    print 'outcomes', infoset.outcome
    #need generic positions
    TerminalNode(last_parent,infoset.outcome)
    return root

class Node(object):
    def __init__(self, parent):
        #self.committed = copy.deepcopy(committed)
        #self.holecards = copy.deepcopy(holecards)
        #self.board = copy.deepcopy(board)
        #self.deck = copy.deepcopy(deck)
        #self.bet_history = copy.deepcopy(bet_history)
        self.children = None
        if parent:
            self.parent = parent
            self.parent.add_child(self)

    def add_child(self, child):
        if self.children is None:
            self.children = [child]
        else:
            self.children.append(child)

    def find(self,parent,x):
        if self.data == x:
            return self
        else:
            return Node(parent,x)

class TerminalNode(Node):
    def __init__(self, parent, outcome):
        Node.__init__(self, parent)
        self.outcome = outcome
        #self.payoffs = payoffs
        #self.players_in = copy.deepcopy(players_in)

class StacksizeNode(Node):
    def __init__(self, parent, stacksize):
            Node.__init__(self, parent)
            self.stacksize = stacksize
            self.children = []
            def find(self,parent,holes,gen_holes):
                for i in xrange(0,len(self.children)):
                    if self.children[i].generic_holecards == gen_holes:
                        return self.children[i]
                return HolecardChanceNode(parent,holes,gen_holes)


class HolecardChanceNode(Node):
    def __init__(self, parent, holecards, generic_holecards):
        Node.__init__(self, parent)
        #self.todeal = todeal
        self.holecards = holecards
        self.generic_holecards = generic_holecards
        self.children = []

        def find(self,parent,x):
            for i in xrange(0,len(self.children)):
                if self.children[i].action == x:
                    return self.children[i]
            return ActionNode(parent,x)

class BoardcardChanceNode(Node):
    def __init__(self, parent, board):
        Node.__init__(self, parent)
        #self.todeal = todeal
        self.board = board
        self.children = []

#TODO add general positions (SB,BB) in addition to Hero?
class ActionNode(Node):
    def __init__(self, parent,action):
        Node.__init__(self, parent)
        #self.player = player
        #self.position = position
        self.action = action
        self.children = []
        self.raise_action = None
        self.bet_action = None
        self.call_action = None
        self.fold_action = None
        #self.player_view = infoset_format(player, holecards[player], board, bet_history)

    def valid(self, action):
        if action == FOLD:
            return self.fold_action
        if action == CALL:
            return self.call_action
        if action == RAISE:
            return self.raise_action
        if action == BET:
            return self.bet_action
        #raise Exception("Unknown action {0}. Action must be FOLD, CALL, or RAISE".format(action))

    def get_child(self, action):
        return self.valid(action)

class BetsizeNode(Node):
    def __init__(self, parent, betsize):
        Node.__init__(self, parent)
        self.betsize = betsize

os.chdir(r"/Users/Shuza/Code/PokerAI/Pickle")
infoset_list = openpickle()
#print infoset_list
infoset = infoset_list[0]
print build_tree(infoset)
