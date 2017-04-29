import numpy as np
import random
import operator
import functools
import Classes as cl

#assign the actual river flush % to suit
def suit(hand):
    suited = 0
    suits = map(lambda x:x[1],hand)
    print suits
    print collections.Counter(suits)
    suitedness = [count for item, count in collections.Counter(suits).items()]
    suitedness.sort(reverse=True)
    print "suited",suitedness
    if suitedness[0] == 1:
        suited = 0
        print "rainbow"
    elif suitedness[0] == 2:
        if suitedness[1] == 2:
            suited = 1
            print "double suited"
        else:
            print "2 single suited"
            suited = .75
    elif suitedness[0] == 3:
        suited = .5
        print "3"
    else:
        suited = .25
        print "mono"
    return suited

#returns how many unique STs a hand can make
#each straight uses 2 cards. So each ST board can be identified by 3 cards
#perhaps start at AKQJT and work downwards? Each board only counts the highest
#ST
def connect(hand):
    print "connect"
    values = map(lambda x:x[0],hand)
    numstraights = 0
    for i in xrange(14,4,-1):
        board = [i,i-1,i-2,i-3,i-4]
        if i == 5:
            board = [14,5,4,3,2]
        print "board",board
        print "hand", values
        print [i for i in board if i in hand]
            #print "yup"
            #numstraights += 1
        print "i",i
    return numstraights

#returns # of nut river hands
def nut(hand):
    return Nuttedness

#returns high card value. multiply the values together? could use prime numbers
#returns value between 0 and 1
def highcard(hand):
    values = map(lambda x:x[0],hand)
    total = functools.reduce(operator.mul, values,1)
    #14^4 is 38416 - 2^4 = 38400 = max-min
    #mean = 3990 approximately
    total = float(total-16) / 38400
    return total

#returns allin vs random hand. uses evan calculator
def equity(hand):
    return equity

#returns the salient aspects of the hand. Removes the explicit suits
def simplify(hand):

    return simple

"""
def findmean(num,deck):
    values = []
    for i in xrange(0,num):
        random.shuffle(deck)
        newhand = deck[:4]
        values.append(highcardstrength(newhand))
    total = (sum(values))/num
    return total
"""

def makeclassfile(deck,combinations):
    hands = list(it.combinations(deck, 2))
    for i in xrange(0,len(hands)):
        hand = simplify(hands[i])
        hand = cl.Rank(hand,suit(hand[i]),connect(hand[i]),highcard(hand[i]),nut(hand[i]),equity(hand[i]))

        #hand.suit = suit(hand[i])
        #hand.connect = connect(hand[i])
        #hand.highcard = highcard(hand[i])
        #hand.nut = nut(hand[i])
        #hand.equity = equity(hand[i])

    #pickle(hands)
    #save file
    return

deck = [[14,'s'],[13,'s'],[12,'s'],[11,'s'],[10,'s'],[9,'s'],[8,'s'],[7,'s'],[6,'s'],[5,'s'],[4,'s'],[3,'s'],[2,'s'],
[14,'h'],[13,'h'],[12,'h'],[11,'h'],[10,'h'],[9,'h'],[8,'h'],[7,'h'],[6,'h'],[5,'h'],[4,'h'],[3,'h'],[2,'h'],
[14,'c'],[13,'c'],[12,'c'],[11,'c'],[10,'c'],[9,'c'],[8,'c'],[7,'c'],[6,'c'],[5,'c'],[4,'c'],[3,'c'],[2,'c'],
[14,'d'],[13,'d'],[12,'d'],[11,'d'],[10,'d'],[9,'d'],[8,'d'],[7,'d'],[6,'d'],[5,'d'],[4,'d'],[3,'d'],[2,'d']]

testhand = [[2,'s'],[2,'c'],[2,'d'],[2,'h']]
anothertest = [[14,'s'],[14,'c'],[14,'d'],[14,'h']]
testy = [[10,'s'],[9,'c'],[8,'d'],[7,'h']]

#print highcard(testhand)

print "hi",type(0.1)
print connect(testy)
#print findmean(100,deck)
