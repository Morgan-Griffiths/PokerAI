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
        #print "Tie"
        return (0,0)
    elif action1 == 0:
        if action2 == 1:
            #print "player1 wins"
            return (1,-1)
        else:
            #print "player2 wins"
            return (-1,1)
    elif action1 == 1:
        if action2 == 0:
            #print "player2 wins"
            return (-1,1)
        else:
            #print "player1 wins"
            return (1,-1)
    elif action1 == 2:
        if action2 == 1:
            #print "player2 wins"
            return (-1,1)
        else:
            #print "player1 wins"
            return (1,-1)

def regret(choice):
    if choice == 0:
        return [1,0,2]
    if choice == 1:
        return [2,1,0]
    if choice == 2:
        return [0,2,1]

#what do i want? all i want, is to take the difference between the utility
#function of our choice and the remaining choices. subtract the utility of our choice
#from the utility of the remaining choices.
def utility(choiceA,choiceB,choices):
    #print 'utility,choices',choiceA,choiceB
    regret_A = d.get((choiceA,choiceB))
    choice_regret = regret_A[0]
    index_A = choices.index(choiceA)
    #print "regret,index,CFV_choice", regret_A, index_A, choice_regret
    regret = [0,0,0]
    for i in xrange(0,3):
        temp = d.get((choices[i],choiceB))
        temp_regret = temp[0]
        regret[i] = temp_regret - choice_regret
        #if regret[i] < 0:
        #    regret[i] = 0
    #print "output",regret
    return regret

def average_strategy():
    avgStrategy = np.zeros(3)
    nomalizing_sum = 0


def cfr(num_iterations):
    choices = [0,1,2]
    #need to calculate the combined regret. This should be done with a matrix
    player1.choice = (random.sample(choices,1))
    player2.choice = (random.sample(choices,1))
    normalizing_sum1 = 0
    normalizing_sum2 = 0
    noderegret = np.zeros((2,3))
    #print noderegret
    for i in xrange(1,num_iterations+1):
        print "choices", player1.choice,player2.choice
        #player_payoffs = payoff(player1.choice,player2.choice)
        player1_payoffs = utility(player1.choice[0],player2.choice[0],choices)
        player2_payoffs = utility(player2.choice[0],player1.choice[0],choices)
        #print "payoffs", player_payoffs
        #print "regret", regret(player2.choice), regret(player1.choice)
        #temp = np.array(regret(player2.choice)) - 1, np.array(regret(player1.choice)) - 1
        temp = np.array(player1_payoffs),np.array(player2_payoffs)
        noderegret[0] = temp[0]
        noderegret[1] = temp[1]
        player1.regret_vector = np.add(player1.regret_vector,player1_payoffs)
        player2.regret_vector = np.add(player2.regret_vector,player2_payoffs)
        player1.cumulative_regret += player1.regret_vector
        player2.cumulative_regret += player2.regret_vector
        print "regret vectors", player1.regret_vector,player2.regret_vector
        constant1 = min(player1.regret_vector)
        if constant1 > 0:
            constant1 = 0
        constant2 = min(player2.regret_vector)
        if constant2 > 0:
            constant2 = 0
        print "constants", constant1,constant2
        temp1 = [player1.regret_vector[x] - constant1 for x in player1.regret_vector]
        temp2 = [player2.regret_vector[x] - constant2 for x in player2.regret_vector]
        print "temps",temp1,temp2
        normalizing_sum1 += sum(temp1) #len of decision array
        normalizing_sum2 += sum(temp2)
        print "normalizing_sum",normalizing_sum1,normalizing_sum2
        player1.action_vector = map(lambda x: float(x)/normalizing_sum1,temp1)
        player2.action_vector = map(lambda x: float(x)/normalizing_sum2,temp2)
        print "newactions", player1.action_vector,player2.action_vector
        chance = random.uniform(0,1)
        player1.choice = np.random.choice([0,1,2],1,p=player1.action_vector)
        player2.choice = np.random.choice([0,1,2],1,p=player2.action_vector)
        #print "newchoice",player1.choice,player2.choice
        #print "Noderegret", noderegret

choices = [0,1,2]
all_actions = [x for x in it.product(choices,repeat=2)]
#print "b",all_actions
all_results = [payoff(i[0],i[1]) for i in all_actions]
#print all_results
d = dict(zip(all_actions,all_results))
#print "d",d
#print d.get((0,1))
#actionpair = random.sample(choices,2)
#testchoice_A = actionpair[0]
#testchoice_B = actionpair[1]
testchoice_A = random.sample(choices,1)
testchoice_B = random.sample(choices,1)
#print utility(testchoice_A[0],testchoice_B[0],choices)



player1 = Player('',[0,0,0],np.zeros(3),0)
player2 = Player('',[0,0,0],np.zeros(3),0)

cfr(100)
