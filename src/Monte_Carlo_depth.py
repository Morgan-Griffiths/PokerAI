import Classes as cl
import Pokereval as pe
import numpy as np
import collections
import random
import math

"""
module for depth first montecarlo. used for random board sampling for hand strength
on flop/turn etc.
Goes to river and then runs sim vs a sample of opponents range?
"""

def depthmontecarlo(herohand,villainrange,board,deck):
    #first we determine how many random cards we need to make
    street = len(board)
    if street == 0:
        #preflop
        #randomly remove 5 cards
    if street == 3:
        #flop
    if street == 4:
        #turn


    for i in xrange(0,len(usedcards)):
        if usedcards[i] in newdeck:
            newdeck.remove(usedcards[i])
    #print "usedcards", usedcards
    #print "deck", newdeck
    #hand1 win, hand2 win, tie, total sims
    results = [0,0,0,0]
    #call the main function
    board = randomboard(newdeck)
    newboard = pe.Board("", "")
    newboard.machinehand = board
    for i in xrange(0,1000):
        #shuffle deck
        board = randomboard(newdeck)
        newboard.machinehand = board
        outcome = pe.main(hand1,hand2,newboard)
        results[3] += 1
        #print "board", board
        #print "outcome", outcome
        if outcome[0] == "Tie!":
            results[2] += 1
        elif outcome[0] == "hand1":
            results[0] += 1
        else:
            results[1] += 1
    #count wins and divide by total sims
    percentage1 = results[0]*100 / results[3]
    percentage2 = results[1]*100 / results[3]
    print "percentage",percentage1,percentage2
    #now need to subtract winning stack from pot and record the amount
    return "results", results, "stacks", player1.stack, player2.stack

#print montecarlo(hero,villain,deck)
