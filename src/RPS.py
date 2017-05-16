import math
import numpy as np
import random
import itertools as it

class Player(object):
    def __init__(self, choice, action_vector, regret_vector, cumulative_regret):
        self.choice = choice
        self.action_vector = action_vector
        self.regret_vector = regret_vector
        self.cumulative_regret = cumulative_regret
Paper = 0
Rock = 1
Sisscors = 2


def payoff(action1,action2):
    if action1 == action2:
        print "Tie"
        return (0,0)
    elif action1 == 0:
        if action2 == 1:
            print "player1 wins"
            return (1,-1)
        else:
            print "player2 wins"
            return (-1,1)
    elif action1 == 1:
        if action2 == 0:
            print "player2 wins"
            return (-1,1)
        else:
            print "player1 wins"
            return (1,-1)
    elif action1 == 2:
        if action2 == 1:
            print "player2 wins"
            return (-1,1)
        else:
            print "player1 wins"
            return (1,-1)

def regret(choice):
    if choice == 0:
        return [1,0,2]
    if choice == 1:
        return [2,1,0]
    if choice == 2:
        return [0,2,1]

def utility(choiceA,choiceB,choices):
    regret_A = d.get(choiceA,choiceB)
    #i = choices.index(choiceA)
    #u = 1 - regret_A
    for i in xrange(0,3):
        regret_sum = d.get(choice(i),choiceB) - regret_A
    return regret_sum

def cfr(num_iterations):
    choices = [0,1,2]
    #need to calculate the combined regret. This should be done with a matrix
    player1.choice = random.sample(choices)
    player2.choice = random.sample(choices)
    noderegret = np.zeros((2,3))
    print noderegret
    for i in xrange(1,num_iterations+1):
        denominator = i*3 #len of decision array
        #print "choices", player1.choice,player2.choice
        #player_payoffs = payoff(player1.choice,player2.choice)
        player_payoffs = d.get((player1.choice,player2.choice))
        #print "payoffs", player_payoffs
        #print "regret", regret(player2.choice), regret(player1.choice)
        temp = np.array(regret(player2.choice)) - 1, np.array(regret(player1.choice)) - 1
        noderegret[0] = temp[0]
        noderegret[1] = temp[1]
        normalizing_sum = []
        player1.regret_vector = np.add(player1.regret_vector,regret(player2.choice))
        player2.regret_vector = np.add(player2.regret_vector,regret(player1.choice))
        player1.action_vector = map(lambda x: float(x/denominator),player1.regret_vector)
        player2.action_vector = map(lambda x: float(x/denominator),player2.regret_vector)
        player1.cumulative_regret += player1.regret_vector
        player2.cumulative_regret += player2.regret_vector
        print "newactions", player1.action_vector,player2.action_vector
        chance = random.uniform(0,1)
        player1.choice = np.random.choice([0,1,2],1,p=player1.action_vector)
        player2.choice = np.random.choice([0,1,2],1,p=player2.action_vector)
        #print "newchoice",player1.choice,player2.choice
        print "Noderegret", noderegret

choices = [0,1,2]
all_actions = [x for x in it.product(choices,repeat=2)]
print "b",all_actions
all_results = [payoff(i[0],i[1]) for i in all_actions]
print all_results
d = dict(zip(all_actions,all_results))
print "d",d
print d.get((0,1))



player1 = Player('',[],np.zeros(3),0)
player2 = Player('',[],np.zeros(3),0)

#cfr(1)
