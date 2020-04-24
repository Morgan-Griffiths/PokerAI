import itertools as it
import numpy as np

"""
generates any relevant omaha hand range as a list
first task. create all g0 hands of any suit
"""

handvalues = {"A" : 14, "K" : 13, "Q" : 12, "J" : 11, "T" : 10,"9" : 9, "8" : 8,
"7" : 7, "6" : 6, "5" :5, "4" : 4, "3" : 3, "2" : 2}

#start from top card and descend. special case for A234
#gather all the necessary cards and do permutations on them
def g0generator(deck):
    cards = []
    temp = []
    for i in xrange(0,2):
        #if i == 8:
            #special case for 4321
        first = list([x for x in deck if x[0] == (14 - i)])
        second = list([x for x in deck if x[0] == (13 - i)])
        third = list([x for x in deck if x[0] == (12 - i)])
        fourth = list([x for x in deck if x[0] == (11 - i)])
        print "allcombos",first,second,third,fourth
        l = list(it.combinations(first,second,third,fourth,4))
        print 'l',l
        #cards.append(l)
    return cards

"""
I want a dictionary or class(?) that contains the following combinations of DECK.
With each combination there is a list of attributes:

5 card combinations: Hand strength
This will allow for quick table lookups for who beat who, and how strong
a given hand is based on the board. Nuttedness feature will use this.

4 card combinations: Hand
Nuttedness - described by the number of nut hands rivered?
Connectedness - described by the number of straights possible
Suitedness - described by the number of flushes possible
High card - quantified by total high card value? subract trips/Quads
Overall equity - quantified by allin vs random hand

Are these necessary? What do they add?
5 card combination: Flop+Turn+river

4 card combinations: Flop/turns

3 card combinations: Flop
"""

#52 card deck
deck = [[14,'s'],[13,'s'],[12,'s'],[11,'s'],[10,'s'],[9,'s'],[8,'s'],[7,'s'],[6,'s'],[5,'s'],[4,'s'],[3,'s'],[2,'s'],
[14,'h'],[13,'h'],[12,'h'],[11,'h'],[10,'h'],[9,'h'],[8,'h'],[7,'h'],[6,'h'],[5,'h'],[4,'h'],[3,'h'],[2,'h'],
[14,'c'],[13,'c'],[12,'c'],[11,'c'],[10,'c'],[9,'c'],[8,'c'],[7,'c'],[6,'c'],[5,'c'],[4,'c'],[3,'c'],[2,'c'],
[14,'d'],[13,'d'],[12,'d'],[11,'d'],[10,'d'],[9,'d'],[8,'d'],[7,'d'],[6,'d'],[5,'d'],[4,'d'],[3,'d'],[2,'d']]

smalldeck = [[14,'s'],[13,'s'],[12,'s'],[11,'s']]

#hands = list(it.combinations(deck, 2))
#print len(hands)
print g0generator(deck)
