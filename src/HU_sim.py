import Pokereval as pe
import Classes as cl
import random
import math
import copy

"""
need to assign variables to hands/flops/actions, so it can tell whats working and whats not.
Make pre and post actions robust for multiple players?
make array of variables for action frequencies and bet/raise sizes.
make softmax function
For later:
Tie the montecarlo function with AI so it can evaluate flops/turns etc.
Ability to set up specific boards/hands and have the bot player from there.
Hand generator, akin to PPT or OR. pass through a set of restrictions and it creates
the hands and stores them. Can be weighted. Can be specific types
for AI:
capture all relevant data for heads up:
action,betsize ratio,pot odds,position,stack size,pot,hand
Should be able to differentiate between our action and opponents action.
Possibly multiple variables for hand class?
suitedness,connectedness,highcard value,paired,nut factor
train it first on random allins. should favor
highcard strength, suitedness, connectedness, or high pairs.
Nut factor would be the frequency of getting a nut hand by the river?(could be flop/turns as well)
"""

#player class
class Player(object):
    def __init__(self, hand, stack, action, position, bettotal, allin):
        self.hand = hand
        self.stack = stack
        self.action = action
        self.position = position
        self.bettotal = bettotal
        self.allin = allin

#used for allin pre
def randomboard(deck):
    #shuffle deck, return the top 5 cards
    board = random.sample(deck,5)
    board.sort(reverse=True)
    return board

#Used for generating flop/turn/river
def randomFTR(deck,street):
    print "randomFTR"
    newdeck = deck[:]
    if street == 1:
    #shuffle deck, return the top 3 cards
        random.shuffle(newdeck)
        board = newdeck[:3]
        board.sort(reverse=True)
        newdeck = newdeck[3:]
        return (board, newdeck)
    if street == 2:
        random.shuffle(newdeck)
        board = newdeck[:1]
        newdeck = newdeck[1:]
        board.sort(reverse=True)
        return (board,newdeck)
    if street == 3:
        random.shuffle(newdeck)
        board = newdeck[:1]
        newdeck = newdeck[1:]
        board.sort(reverse=True)
        return (board,newdeck)
#gives player a random hand
def randomhand(name,deck):
    random.shuffle(deck)
    newhand = deck[:4]
    newhand.sort(reverse=True)
    name.hand.cards = newhand
    newdeck = deck[4:]
    return (newhand, newdeck)

#main function - inits players, inits reduced deck to use in sims
def main(deck,stacksize):
    #setting position 0 = btn, 1 = bb
    player1 = Player(None,'','U',0,0, False)
    player2 = Player(None,'','U',1,0, False)
    player1.hand = pe.Hand('', None, [],[])
    player2.hand = pe.Hand('', None, [],[])
    #player1 wins, player2 wins, tie, total games.
    results = [0,0,0,0]
    #player1 winnings, player2 winnings.
    #Pot - bettotal = winnings on that street
    #pot + stack - initial stack = total winnings
    winnings = [0,0]
    for i in xrange(0,1000):
        newdeck = deck[:]
        (player1.hand.cards, newdeck) = randomhand(player1, newdeck)
        (player2.hand.cards, newdeck) = randomhand(player2, newdeck)
        player1.hand.machinehand = copy.deepcopy(player1.hand.cards)
        player2.hand.machinehand = copy.deepcopy(player2.hand.cards)
        print 'init hands',player1.hand.machinehand, player2.hand.machinehand
        outcome, action, pot = minigame(player1,player2,newdeck,stacksize)
        print "outcome", outcome
        results[3] += 1
        if outcome == "Tie!":
            results[2] += 1
        elif outcome == "hand1":
            results[0] += 1
            print "stuff1", (player1.stack, pot)
            amnt = player1.stack + pot
            loss = player2.stack - stacksize
            winnings[0] += (amnt - stacksize)
            winnings[1] += (loss)
        else:
            results[1] += 1
            print "stuff2", (player2.stack, pot)
            amnt = player2.stack + pot
            loss = player1.stack - stacksize
            winnings[1] += (amnt - stacksize)
            winnings[0] += (loss)
    return results, outcome, action,"pot", pot, "winnings", winnings

#minigame function
def minigame(player1,player2,deck,stacksize):
    player1.stack = stacksize
    player2.stack = stacksize
    street = 0
    bigblind = 2
    smallblind = 1
    pot = bigblind + smallblind
    player1.stack -= smallblind
    player2.stack -= bigblind
    player1.bettotal = smallblind
    player2.bettotal = bigblind
    player1.allin = False
    player2.allin = False
    #each sub list in action has 3 attributes. Action, Betsize(if any), ratio to pot(if any)
    #seed the blinds into action list
    action = [['SB',1,0],['BB',2,0]]
    board = pe.Board('','')
    newdeck = deck[:]
    if street == 0:
        (outcome, action, board, pot, newdeck, street) = preflop(player1,player2,street,newdeck,pot,board,action,smallblind,bigblind)
        if action[-1][0] == 'F':
            return outcome, action, pot
        street = street + 1
    while street < 4:
        #reset total investments per street pre player
        print "street",street
        player1.bettotal = 0
        player2.bettotal = 0
        #outcome and action are doing double duty here, could get rid of outcome and change the result evaluator
        (outcome, action, board, pot, newdeck, street) = postflop(player1,player2,street,newdeck,pot,board,action,smallblind,bigblind)
        if action[-1][0] == 'F':
            break
        street = street + 1
        print "action", action
        print "stacks,pot", player1.stack, player2.stack,pot
    if street == 4 and action[-1][0] == 'C' or action[-2][0] == 'X' and action[-1][0] == 'X':
        print "showdown!"
        #board = randomboard(deck)
        #newboard = pe.Board("", "")
        #newboard.machinehand = board
        print "player1", player1.hand.machinehand, "player2", player2.hand.machinehand, "board", board.machinehand
        board.machinehand.sort(reverse=True)
        return pe.main(player1.hand,player2.hand,board), action, pot
    else:
        print "hand over"
        return outcome, action, pot

#keeping track of actions - preflop-extended preflop,postflop-extended postflop
#sets up pot preflop
#calls function to determine player actions, which it stores in player.action
#returns outcome, action, board(if any), pot, newdeck
#need a flag for an allin player, so they can't make any more actions
def preflop(player1,player2,street,newdeck,pot,board,action,smallblind,bigblind):
    print "preflop"
    #first action
    action.append(['C',1,0.25])
    player1.stack -= 1
    (player1.action, action, pot) = 'C',action,4#preaction(player1,action,smallblind,pot)
    print "player1.action", player1.action
    if player1.action == 'F':
        print "player1 folded"
        return "hand2", action, board, pot, newdeck, street
    #special case for C
    action.append(['X',0,0])
    (player2.action, action, pot) = 'X',action,4#prereaction(player2,action,bigblind,pot)
    print "player2.action", player2.action
    print "action", action
    if action[-2][0] == 'R' and action[-1][0] == 'C' or action[-2][0] == 'C' and action[-1][0] == 'X':
        return "next street", action, board, pot, newdeck, street
    if player2.action == 'F':
        print "player2 folded"
        return "hand1", action, board, pot, newdeck, street
    i = 0
    #position = num players
    position = 1
    while i != position:
        print "enterwhile"
        print "stacks", player1.stack,player2.stack
        #Check if player is allin
        if player1.allin == False:
            (player1.action, action, pot) = randomaction(player1,player2,action,bigblind,pot)
            if player1.action == 'F':
                print "player1.action", player1.action
                print "action", action
                return "hand2", action, board, pot, newdeck, street
        #Check if player is allin
        if player2.allin == False:
            (player2.action, action, pot) = randomaction(player2,player1,action,bigblind,pot)
            print "post player2.action", player2.action
        if player1.action == 'R':
            position = 0
            i = 1
        if player2.action == 'R':
            position = 1
            i = 0
        else:
            print "i",i,"position",position
            position = i
        print "action", action
        print "stacks", player1.stack,player2.stack
        #check if anyone folded
        if player2.action == 'F':
            print "player2 folds"
            return "hand1", action, board, pot, newdeck, street
        if player1.allin == True and player2.allin == True:
            #finish runout and return winner etc to minigame
            return runout(street,board,action,player1,player2,newdeck,pot)
    #check if both players allin

#postflop
def postflop(player1,player2,street,newdeck,pot,board,action,smallblind,bigblind):
    #instantiating flop/turn/river
    if street == 1:
        #flop
        (flop, newdeck) = randomFTR(newdeck,1)
        board.machinehand = flop
        board.machinehand.sort(reverse=True)
        print "flop", board.machinehand
    if street == 2:
        #turn
        (turn, newdeck) = randomFTR(newdeck,2)
        print "turn", turn
        print "board.machinehand", board.machinehand
        board.machinehand += turn
        print "+turn", board.machinehand
    if street == 3:
        #river
        (river, newdeck) = randomFTR(newdeck,3)
        board.machinehand += river
        print "+river", board.machinehand
    print "postflop"
    print "stacks", player1.stack,player2.stack
    print "not allin",player2.allin == False
    print "not allin",player1.allin == False
    #first action
    (player2.action, action, pot) = postaction(player2, action, bigblind, pot)
    (player1.action, action, pot) = randomaction(player1,player2, action, bigblind, pot)
    print "post player2.action", player2.action
    print "post player1.action", player1.action
    print "actions", action
    print "pot", pot
    if action[-1][0] == 'C' or action[-2][0] == 'X' and action[-1][0] == 'X':
        return "next street", action, board, pot, newdeck, street
    if player1.action == 'F':
        return "hand2", action, board, pot, newdeck, street
    if player2.action == 'F':
        return "hand1", action, board, pot, newdeck, street
    i = 0
    #position = num players
    position = 1
    while i != position:
        print "enterpostwhile"
        print "stacks", player1.stack,player2.stack
        #Check if player1 is allin
        print "not allin",player2.allin == False
        print "not allin",player1.allin == False
        if player2.allin == False:
            print "Player2 action"
            (player2.action, action, pot) = randomaction(player2, player1, action, bigblind, pot)
        #Check if player2 is allin
        if player1.allin == False:
            print "Player1 action"
            (player1.action, action, pot) = randomaction(player1, player2, action, bigblind, pot)
        if player2.action == 'R':
            position = 0
        if player1.action == 'R':
            position = 1
        else:
            position = i
        print "post player2.action", player2.action
        print "post player1.action", player1.action
        print "actions", action
        print "stacks", player1.stack,player2.stack
        print "i, position",i, position
        #check if anyone folded
        if player1.action == 'F':
            print "player1 fold"
            return "hand2", action, board, pot, newdeck, street
        if player2.action == 'F':
            print "player2 fold"
            return "hand1", action, board, pot, newdeck, street
        if player1.allin == True and player2.allin == True:
            print "allin"
            #finish runout and return winner etc to minigame
            if street < 3:
                return runout(street,board,action,player1,player2,newdeck, pot)
            else:
                return "Showdown", action, board, pot, newdeck, street
    return "next street", action, board, pot, newdeck, street

#Action generation functions
#preflop HU actions
#already subtracted the blinds from the stacks
def preaction(player,action,smallblind,pot):
    print "preaction", action
    a = ['R','C','F']
    choice = random.choice(a)
    print 'choice',choice
    if choice == 'C':
        #subtract call
        player.stack -= smallblind
        player.bettotal += smallblind
        pot += smallblind
        ratio = float(smallblind)/pot
        ratio = math.ceil(ratio * 100) / 100
        action.append([choice, smallblind, ratio])
        #returns the action to the player
        return (choice,action,pot)
    if choice == 'F':
        #subtract nothing
        action.append([choice,0,0])
        return (choice,action,pot)
    if choice == 'R':
        return sbraisepre(player,choice,action,smallblind,pot)
def prereaction(player,action,bigblind,pot):
    print 'prereaction', action
    if action[-1][0] == 'C':
        a = ['X','R']
        choice = random.choice(a)
        if choice == 'X':
            action.append([choice, 0, 0])
            return (choice,action,pot)
        if choice == 'R':
            #special case for preflop raises. should be treated equally
            return bbraisepre(player,choice,action,bigblind,pot)
    if action[-1][0] == 'R':
        a = ['F','C','R']
        choice = random.choice(a)
        if choice == 'F':
            action.append([choice,0,0])
            return (choice,action,pot)
        if choice == 'C':
            #subtract betsize
            #callbet returns new stack, pot
            return call(player,choice,action,pot)
        if choice == 'R':
            return raised(player,choice,action,pot)

    #action == 'F'
    else:
        choice = 'W'
        action.append([choice,0,0])
        return (choice,action,pot)
#first action postflop
def postaction(player,action,bigblind,pot):
    print "postaction", action
    a  = ['B','X']
    choice = random.choice(a)
    if choice == 'B':
        return bettor(player,choice,action,bigblind,pot)
    if choice == 'X':
        action.append([choice,0,0])
        return choice, action, pot

#subsequent actions. Need to check when facing B or R that villain isn't allin,
#determines action for playerA, checks to make sure playerB isn't allin
def randomaction(playerA,playerB,action,bigblind,pot):
    print "randomaction", action
    print "action", action[-1][0]
    print "stacks", playerA.stack, playerB.stack
    if action[-1][0] == 'B':
        if playerB.stack == 0:
            a = ['F','C']
            choice = random.choice(a)
        else:
            a = ['F','C','R']
            choice = random.choice(a)
        if choice == 'F':
            action.append([choice,0,0])
            return choice,action,pot
        if choice == 'C':
            return call(playerA,choice,action,pot)
        if choice == 'R':
            return raised(playerA,choice,action,pot)
    if action[-1][0] == 'R':
        if playerB.stack == 0:
            a = ['F','C']
            choice = random.choice(a)
        else:
            a = ['F','C','R']
            choice = random.choice(a)
        if choice == 'F':
            action.append(['F',0,0])
            return choice,action,pot
        if choice == 'C':
            return call(playerA,choice,action,pot)
        if choice == 'R':
            return raised(playerA,choice,action,pot)
    if action[-1][0] == 'X':
        a = ['X','B']
        choice = random.choice(a)
        if choice == 'X':
            action.append(['X',0,0])
            return choice,action,pot
        if choice == 'B':
            return bettor(playerA,choice,action,bigblind,pot)
    if action[-1][0] == 'F':
        #end of hand
        return '',action,pot
    if action[-1][0] == 'C':
        #next street
        return '',action,pot

#streamlined functions
#Adds betsize to pot. subtracts betsize from player.stack. adds betsize to player.bettotal.
#actions,betsize,ratio to action. Returns choice to player.action
def call(player,choice,action,pot):
    #should add a check to see if player.stack < betsize. when introducing variable stacks and bettor gets a discount
    print "call"
    betsize = copy.copy(action[-1][1])
    print "initbet", betsize
    betsize -= player.bettotal
    print "player.bettotal", player.bettotal
    print "betsize", betsize
    player.bettotal += betsize
    player.stack -= betsize
    ratio = float(betsize) / (pot+betsize)
    ratio = math.ceil(ratio * 100) / 100
    pot += betsize
    action.append([choice,betsize,ratio])
    if player.stack == 0:
        player.allin = True
    return choice,action,pot
def bettor(player,choice,action,bigblind,pot):
    print "bettor"
    betsize = randombet(player,bigblind,pot)
    player.bettotal += betsize
    player.stack -= betsize
    ratio = float(betsize)/pot
    ratio = math.ceil(ratio * 100) / 100
    pot += betsize
    action.append([choice,betsize,ratio])
    if player.stack == 0:
        player.allin = True
    return choice,action,pot
def raised(player,choice,action,pot):
    print "raised"
    print "stack", player.stack
    (raisesize, maxraise) = randomraise(player,action,pot)
    temp = copy.copy(raisesize)
    player.stack -= raisesize
    ratio = float(raisesize)/maxraise
    ratio = math.ceil(ratio * 100) / 100
    pot += raisesize
    raisesize += player.bettotal
    player.bettotal += temp
    action.append([choice,raisesize,ratio])
    if player.stack == 0:
        player.allin = True
    return choice,action,pot

#possible actions
#B R C X F
#B - R,C,F | R - R C F | C - R,X or next card | X - B X | F hand over | Unopened 'U' - B,X

#special case for BB facing sb call and Sb raise. Adds raisesize to pot, subtracts raisesize from stack
#adds raisesize to bettotal. Add everything to action. calc ratio
def sbraisepre(player,choice,action,smallblind,pot):
    print "sbraisepre"
    maxraise = 5*smallblind
    minraise = 3*smallblind
    raisesize = random.randint(minraise,maxraise)
    player.stack -= raisesize
    player.bettotal += raisesize
    pot += raisesize
    ratio = float(raisesize)/maxraise
    ratio = math.ceil(ratio * 100) / 100
    raisesize += smallblind
    action.append([choice,raisesize,ratio])
    return choice,action,pot
def bbraisepre(player,choice,action,bigblind,pot):
    print "bbraisepre"
    maxraise = 2*bigblind
    minraise = bigblind
    raisesize = random.randint(minraise,maxraise)
    player.stack -= raisesize
    player.bettotal += raisesize
    pot += raisesize
    ratio = float(raisesize)/maxraise
    ratio = math.ceil(ratio * 100) / 100
    #in order to make the raisesize absolute
    raisesize += bigblind
    action.append([choice,raisesize,ratio])
    return choice,action,pot

#creates a random betsize between bigblind and pot,
def randombet(player,bigblind,pot):
    print "randombet"
    if player.stack > pot:
        betsize = random.randint(bigblind,pot)
        return betsize
    if player.stack <= pot and player.stack >= bigblind:
        betsize = random.randint(bigblind,player.stack)
        return betsize
    else:
        betsize = player.stack
        return betsize

#creates a random raise size. subtracts raisesize from stack. adds raisesize to bettotal
def randomraise(player,action,pot):
    print "randomraise"
    print "stacks", player.stack
    #pot raise = pot+bet*3
    vilbet = action[-1][1]
    bet = vilbet - player.bettotal
    maxraise = pot+bet+vilbet
    #min raise = 2x vilbet
    minraise = bet*2
    print "minraise, absBet",minraise, bet
    if player.stack > maxraise:
        raisesize = random.randint(minraise,maxraise)
        return raisesize,maxraise
    if player.stack <= maxraise and player.stack >= minraise:
        raisesize = random.randint(minraise,player.stack)
        return raisesize,maxraise
    #if stack is less than the bigblind
    else:
        raisesize = player.stack
        return raisesize,maxraise

#for pre river allins. Set street to 3 before return.
#(outcome, action, board, pot, newdeck)
def runout(street,board,action,player1,player2,deck,pot):
    print "runout"
    print (street,board,action,player1,player2,pot)
    if street == 0:
        #allin pre
        board.machinehand = randomboard(deck)
        street = 3
        return "allin 0",action,board,pot,deck,street
    if street == 1:
        #append turn/river to board
        turn,newdeck = randomFTR(deck,2)
        river,newdeck = randomFTR(newdeck,3)
        board.machinehand += turn
        board.machinehand += river
        street = 3
        return "allin 1", action,board,pot,newdeck,street
    if street == 2:
        #append river to board
        river,newdeck = randomFTR(deck,3)
        board.machinehand += river
        street = 3
        return "allin 2", action,board,pot,newdeck,street

#52 card deck
deck = [[14,'s'],[13,'s'],[12,'s'],[11,'s'],[10,'s'],[9,'s'],[8,'s'],[7,'s'],[6,'s'],[5,'s'],[4,'s'],[3,'s'],[2,'s'],
[14,'h'],[13,'h'],[12,'h'],[11,'h'],[10,'h'],[9,'h'],[8,'h'],[7,'h'],[6,'h'],[5,'h'],[4,'h'],[3,'h'],[2,'h'],
[14,'c'],[13,'c'],[12,'c'],[11,'c'],[10,'c'],[9,'c'],[8,'c'],[7,'c'],[6,'c'],[5,'c'],[4,'c'],[3,'c'],[2,'c'],
[14,'d'],[13,'d'],[12,'d'],[11,'d'],[10,'d'],[9,'d'],[8,'d'],[7,'d'],[6,'d'],[5,'d'],[4,'d'],[3,'d'],[2,'d']]

#hero = pe.Hand("8s9d2c3c", None, [],[])
#villain = pe.Hand("3s4dKh7c", None, [],[])
#board = pe.Board("ThTd6s7s5s", None)

print main(deck,100)
#print randomboard(deck)
#print pe.main(hero,villain,board)
