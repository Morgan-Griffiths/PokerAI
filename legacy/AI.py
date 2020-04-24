"""
This is the module for testing learning strategies.
cost function: 1/2m * sum 1 to M(h(x^i) - y^i)^2
"""
import Pokereval as pe
import Classes as cl
import numpy as np
import random
import collections

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)
def newsoftmax(x):
    return 1.0 / (1.0 + np.exp(-x))



#generate random hands for each player, plus random board. return outcome.
#if AI loses -1 wins +1 after sims it returns a hand ranking.
def randomhand(name,deck):
    random.shuffle(deck)
    newhand = deck[:4]
    newhand.sort(reverse=True)
    name.hand.cards = newhand
    newdeck = deck[4:]
    return (name, newdeck)

def randomboard(deck):
    #shuffle deck, return the top 5 cards
    board = random.sample(deck,5)
    board.sort(reverse=True)
    return board

def connectedness():
    return connected

#suitedness ranking
#rainbow,mono,3,2,ds
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

def outcome(hand1,hand2,deck):
    results = [0,0,0,0]
    board = cl.Board("", "")
    board.machinehand = randomboard(deck)
    winner = pe.main(hand1,hand2,board)
    print winner
    if winner[0] == "Tie!":
        return 0.5
    elif winner[0] == "hand1":
        return 1
    else:
        return 0

def guess(suitedness,a,b):
    #linear function ax+b
    x = suitedness
    value = a * x + b
    return newsoftmax(value)

"""
Omega j := Omega j - Alpha(learning rate) * del / del omega j * cost (omega 0,omega 1) for j = 0 and j = 1
Simultaneously update theta 0 and theta 1. Compute both values and update both values
"""
def update(guess,a,b,s,loss,result,interval):
    learningrate = 0.005
    print "guess,loss,result", guess, loss, result
    newloss = sum(loss) / interval
    #deriv1 1/m * sum of (ax+b)-y * x
    tempa = a - learningrate * derivative of a * newloss
    #deriv2 1/m * sum (ax+b)-y
    tempb = b - learningrate * derivative of b * newloss
    newa = tempa
    newb = tempb
    print "new values",newa,newb
    return newa,newb

def learn(num,interval):
    player1 = cl.Player('','','',0,0,False)
    player1.hand = cl.Hand('', None, [],[])
    player2 = cl.Player('','','',0,0,False)
    player2.hand = cl.Hand('', None, [],[])
    a = random.randint(-10,10)
    b = random.randint(-10,10)
    loss = []
    totalloss = []
    print "init values", a,b
    for i in xrange(0,num):
        player1, newdeck = randomhand(player1,cl.DECK)
        player1.hand.machinehand = player1.hand.cards
        player2, newdeck = randomhand(player2,newdeck)
        player2.hand.machinehand = player2.hand.cards
        print "the hands",player1.hand.cards, player2.hand.cards
        s = suit(player1.hand.machinehand)
        g = guess(s,a,b)
        print "guess", g
        result = 1#outcome(player1.hand,player2.hand,newdeck)
        loss.append((g-result)**2)
        if i % interval == 0:
            (a,b) = update(g,a,b,s,loss,result,interval)
            totalloss.append(loss)
            loss = []

    return a,b


print learn(200,10)

#print train(1,deck)

#scores = [3.0, 1.0, 0.2]
#print(softmax(scores))
