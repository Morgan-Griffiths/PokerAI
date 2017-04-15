import numpy as np
import itertools
import collections


"""
Omaha hand evaluator
Case 1 High card
Case 2 Pair
Case 3 2pair
Case 4 Trips
Case 5 Straight
Case 6 Flush
Case 7 Full house
Case 8 Quads
Case 9 Straight flush
"""

#dictionary
handvalues = {"A" : 14, "K" : 13, "Q" : 12, "J" : 11, "T" : 10,"9" : 9, "8" : 8,
"7" : 7, "6" : 6, "5" :5, "4" : 4, "3" : 3, "2" : 2}

#Hand class to be used for updating river hand strength
class Hand(object):

    def __init__(self, cards, machinehand, riverstrength, rivervalues):
        self.cards = cards
        self.machinehand = machinehand
        self.riverstrength = riverstrength
        self.rivervalues = rivervalues

#Similar class for Board (maybe redundant)
class Board(object):
    def __init__(self, cards, machinehand):
        self.cards = cards
        self.machinehand = machinehand

#Converting human readable hands into machine readable lists of lists
def cardconversion(cards):
    #newcards = " ".join(cards[i:i+2] for i in range(0, len(cards), 2))
    newcards = list(cards)

    print "newcards", newcards
    newcards = [newcards[i:i+2] for i in range(0,len(newcards),2)]
    print "newcards", newcards
    for i in xrange(0,len(newcards)):
         #print handvalues.get(newcards[i][0])
         newcards[i][0] = handvalues.get(newcards[i][0])
    #newcards = newcards[np.argsort(newcards[:,0])]
    newcards.sort(reverse = True)
    return newcards

#iterates through all hand and board combinations - calls evaluation
def handstrength(name,hand,board):
    absolutestrength = [[0],[0]]
    #iterate through all the hero's cards
    for i in xrange(0,len(hand)):
        firstcard = hand[i]
        for j in xrange(i+1, len(hand)):
            secondcard = hand[j]
            #print firstcard,secondcard
            testhand = firstcard,secondcard
            #now we have the hand. now break the board into 3card chunks
            for x in xrange(0,len(board)):
                firstboard = board[x]
                for y in xrange(x+1,len(board)):
                    secondboard = board[y]
                    for z in xrange(y+1, len(board)):
                        thirdboard = board[z]
                        combinedb = firstboard, secondboard, thirdboard
                        #print combinedb
                        #now we find the 5card hand strength
                        allcards = testhand + combinedb
                        allcards = list(allcards)
                        allcards.sort(reverse = True)
                        #print "allcards", allcards
                        currentstrength = evaluation(name,allcards)
                        #print "currentstrength", currentstrength[0],currentstrength[1]
                        if currentstrength[0][0] > absolutestrength[0][0]:
                            #need to solve for ties
                            absolutestrength = currentstrength
                            #print "post absolutestrength", absolutestrength[1]
                        elif currentstrength[0][0] == absolutestrength[0][0]:
                            #print "Going to tiebreaker"
                            outcome = tiebreaker(currentstrength,absolutestrength)
                            if outcome[0] == "hand1":
                                absolutestrength = currentstrength
                        #print "absolutestrength", absolutestrength
    return absolutestrength

#returns handstrength - case number + 5 card hand.
def evaluation(name,cards):
    rivercase = ""
    comparison = [item[0] for item in cards]
    suits = [item[1] for item in cards]
    #print "suits", suits
    #print "comparison", comparison
    #test if flush possible
    if suits.count(suits[0]) == len(suits):
        #print "haveflush"
        #have flush, now test for straight flush
        if all(earlier == later+1 for earlier, later in zip(comparison, comparison[1:])) or comparison == [14,5,4,3,2]:
            #print "straightflush"
            rivercase = "straightflush"
            name.riverstrength = [9,cards]
            name.rivervalues = comparison
        else:
            #print "flush"
            rivercase = "flush"
            name.riverstrength = [6,cards]
            name.rivervalues = comparison
    #test for straight
    elif all(earlier == later+1 for earlier, later in zip(comparison, comparison[1:])) or comparison == [14,5,4,3,2]:
        #print "straight"
        rivercase = "straight"
        name.riverstrength = [5,cards]
        name.rivervalues = comparison
    #test for pairs quads etc.
    else:
        index = 0
        while index < len(cards)-1:
            value = cards[index][0]
            #function cardcount counts duplicates
            num = cardcount(cards,value)
            #print "num,value", num, value
            if num > 1:
                if num > 3:
                    #print "foundquads"
                    rivercase += "quads"
                    name.riverstrength = [8,cards]
                    name.rivervalues = comparison
                if num > 2:
                    rivercase += "trips"
                    #print name.riverstrength
                else: #if num = 2
                    rivercase += "pair"
                    #print name.riverstrength
            index = index + num
        if rivercase == "pairpair":
            name.riverstrength = [3,cards]
            name.rivervalues = comparison
        elif rivercase == "tripspair" or rivercase == "pairtrips":
            name.riverstrength = [7,cards]
            name.rivervalues = comparison
        elif rivercase == "trips":
            name.riverstrength = [4,cards]
            name.rivervalues = comparison
        elif rivercase == "pair":
            name.riverstrength = [2,cards]
            name.rivervalues = comparison
        elif rivercase == "":
            name.riverstrength = [1,cards]
            name.rivervalues = comparison
        #print "End"
    return name.riverstrength,name.rivervalues

def cardcount(cards,value):
    count = 0
    for i in xrange(0, len(cards)):
        #print 'value',cards[i][0]
        #print 'count',count
        if value == cards[i][0]:
            count = count + 1
    return count

#determines the outcome. Win, Lose or Tie. Calls handstrength
def winner(hand1,hand2):
    if hand1[0][0] != hand2[0][0]:
       if hand1[0][0] > hand2[0][0]:
           return "hand1",hand1
       else:
           return "hand2",hand2
    else:
        #print "breaking tie..."
        return tiebreaker(hand1,hand2)

def tiebreaker(hand1,hand2):
    intersection = set(hand1[1]).intersection(hand2[1])
    #print "intersection", intersection
    if hand1[1] == hand2[1]:
        return "Tie!"
    #1 - high card
    if hand1[0][0] == 1:
        index = 0
        while hand1[index][1] == hand2[index][1] and index <= 4:
            index = index + 1
        if hand1[index] > hand2[index]:
            return "hand1",hand1
        else:
            return "hand2",hand2
    #2 - Pair
    if hand1[0][0] == 2:
        #print "pair"
        #if one repeating card is >, they win. else go down the list of nonerepeats
        hand1repetitions = [item for item, count in collections.Counter(hand1[1]).items() if count > 1]
        hand1repetitions.sort(reverse = True)
        #print "hand1repetitions", hand1repetitions
        hand2repetitions = [item for item, count in collections.Counter(hand2[1]).items() if count > 1]
        hand2repetitions.sort(reverse = True)
        #print "hand2repetitions", hand2repetitions
        #if the pair is unequal
        if hand1repetitions != hand2repetitions:
            if hand1repetitions > hand2repetitions:
                return "hand1",hand1
            else:
                return "hand2",hand2
        #pair is equal, look at the other cards.
        else:
            #subtract pair from arrays and compare remaining cards
            player1 = [value for value in hand1[1] if value != hand1repetitions[0]]
            player2 = [value for value in hand2[1] if value != hand2repetitions[0]]
            #compare remaining hards
            index = 0
            while player1[index] == player2[index] and index <= 4:
                index = index + 1
            if player1[index] > player2[index]:
                return "hand1",hand1
            else:
                return "hand2",hand2

    #3 - 2pair
    if hand1[0][0] == 3:
        #print "2 pair"
        #if one repeating card is >, they win. else go down the list of nonerepeats
        hand1repetitions = [item for item, count in collections.Counter(hand1[1]).items() if count > 1]
        hand1repetitions.sort(reverse = True)
        #print "hand1repetitions", hand1repetitions
        hand2repetitions = [item for item, count in collections.Counter(hand2[1]).items() if count > 1]
        hand2repetitions.sort(reverse = True)
        #print "hand2repetitions", hand2repetitions
        #if the pair is unequal
        i = 0
        while (i < 2 and hand1repetitions[i] == hand2repetitions[i]):
            i = i + 1
            if i == 2:
                i = 1
                break
        if hand1repetitions[i] != hand2repetitions[i]:
            if hand1repetitions[i] > hand2repetitions[i]:
                return "hand1",hand1
            else:
                return "hand2",hand2
        #check the highest duplicate, check next duplicate. Else 5th card
        else:
            #subtract the pairs from arrays and compare remaining cards
            #print "hand1[1]", hand1[1]
            #print "hand2[1]", hand2[1]
            #print hand1repetitions
            player1 = [value for value in hand1[1] if value != hand1repetitions[0] and value != hand1repetitions[1]]
            player2 = [value for value in hand1[1] if value != hand1repetitions[0] and value != hand1repetitions[1]]
            #print "player1", player1
            #compare the remaining card
            if player1[0] > player2[0]:
                return "hand1",hand1
            else:
                return "hand2",hand2
    #4 - Trips
    if hand1[0][0] == 4:
        #check trips, check next cards
        #print "trips"
        hand1repetitions = [item for item, count in collections.Counter(hand1[1]).items() if count > 1]
        #print "hand1repetitions", hand1repetitions
        hand2repetitions = [item for item, count in collections.Counter(hand2[1]).items() if count > 1]
        #print "hand2repetitions", hand2repetitions
        if hand1repetitions != hand2repetitions:
            if hand1repetitions > hand2repetitions:
                return "hand1",hand1
            else:
                return "hand2",hand2
        #subtract trips, look at the other cards.
        else:
            #subtract the pairs from arrays and compare remaining cards
            player1 = [value for value in hand1[1] if value != hand1repetitions[0]]
            player2 = [value for value in hand2[1] if value != hand2repetitions[0]]
            #compare the remaining card
            if player1[0] > player2[0]:
                return "hand1",hand1
            else:
                return "hand2",hand2
    #5 - Straight
    if hand1[0][0] == 5:
        #print "straight"
        #check top card (wheel exception)
        if hand1[0][1] == 14 and hand1[1][1] == 5:
            return "hand2",hand2
        elif hand2[0][1] == 14 and hand1[1][1] == 5:
            return "hand1",hand1
        elif hand1[0][1] > hand2[0][1]:
            return "hand1",hand1
        else:
            return "hand2",hand2
    #6 - Flush
    if hand1[0][0] == 6:
        #print "flush"
        #check all cards
        index = 0
        while hand1[index][1] == hand2[index][1] and index <= 4:
            index = index + 1
        if hand1[index][1] > hand2[index][1]:
            return "hand1",hand1
        else:
            return "hand2",hand2
    #7 - Full house
    if hand1[0][0] == 7:
        #print "fullhouse"
        hand1repetitions = [item for item, count in collections.Counter(hand1[1]).items() if count > 2]
        #print "hand1repetitions", hand1repetitions
        hand2repetitions = [item for item, count in collections.Counter(hand2[1]).items() if count > 2]
        #print "hand2repetitions", hand2repetitions
        if hand1repetitions != hand2repetitions:
            if hand1repetitions > hand2repetitions:
                return "hand1",hand1
            else:
                return "hand2",hand2
        #trips is equal, remove trips from array and look at the pair.
        else:
            #print "breaking tie by pair"
            player1 = [value for value in hand1[1] if value != hand1repetitions[0]]
            player2 = [value for value in hand2[1] if value != hand2repetitions[0]]
            #print player1,player2
            if player1 > player2:
                return "hand1",hand1
            else:
                return "hand2",hand2
    #8 - Quads
    if hand1[0][0] == 8:
    #check quads, else tie (not possible in omaha)
        #print "quads"
        hand1repetitions = [item for item, count in collections.Counter(hand1[1]).items() if count > 1]
        hand2repetitions = [item for item, count in collections.Counter(hand2[1]).items() if count > 1]
        if hand1repetitions != hand2repetitions:
            if hand1repetitions > hand2repetitions:
                return "hand1",hand1
            else:
                return "hand2",hand2
        index = 0
        #need to subtract quads from array
        player1 = [value for value in hand1[1] if value != hand1repetitions[0]]
        player2 = [value for value in hand2[1] if value != hand2repetitions[0]]
        while hand1[index][1] == hand2[index][1] and index <= 4:
            index = index + 1
        if hand1[index] > hand2[index]:
            return "hand1",hand1
        else:
            return "hand2",hand2
    #9 - Straight flush
    else:
        #check top card (wheel exception first)
        #print "straightflush"
        if hand1[0][1] == 14 and hand1[1][1] == 5:
            return "hand2",hand2
        elif hand2[0][1] == 14 and hand1[1][1] == 5:
            return "hand1",hand1
        elif hand1[0][1] > hand2[0][1]:
            return "hand1",hand1
        else:
            return "hand2",hand2

def main(name1,name2,board):
    #name1.machinehand = cardconversion(name1.cards)
    #name2.machinehand = cardconversion(name2.cards)
    #board.machinehand = cardconversion(board.cards)
    hand1 = handstrength(name1,name1.machinehand,board.machinehand)
    hand2 = handstrength(name2,name2.machinehand,board.machinehand)
    print "hand1", hand1, "hand2", hand2
    return winner(hand1,hand2)


#Initialize variables
wheel = Hand("As2d3h4s", None, [],[])
hero = Hand("8s9d2c3c", None, [],[])
villain = Hand("3s4dKh7c", None, [],[])
board = Board("ThTd6s7s5s", None)
anotherboard = Hand("Ad2c8c3d5d", None, [],[])

#print hero,villain,board
#print main(hero,villain,board)

"""
#Adding machine readable hands
board.machinehand = cardconversion(board.cards)
wheel.machinehand = cardconversion(wheel.cards)
hero.machinehand = cardconversion(hero.cards)
villain.machinehand = cardconversion(villain.cards)
anotherboard.machinehand = cardconversion(anotherboard.cards)

#testing specific hands
hand1 = handstrength(hero,hero.machinehand,board.machinehand)
hand2 = handstrength(villain,villain.machinehand,board.machinehand)
print "hand1",hand1
print "hand2",hand2
print "who wins?", winner(hand1,hand2)
"""
