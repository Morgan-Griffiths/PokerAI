import math

"""
POMDP = (S,A,T,R,Omega,O,Gamma)
S = set of all states
A = set of all actions
T = transition probabilities between states
R = rewards for transitions between states S * A -> R
Omega = set of observations
O = set of conditional observed probabilities
Gamma = discount factor

At each time period, agent takes some action A in some state S, which causes
the environment to transition with probability T(s'|s,a). Agent also receives
an observation Omega which depends on the new state of the environment with
probability O(o|s',a). Then the agent receives a reward R(s,a). Repeat.

Goal is to maximize the discount reward E

E = sum(gamma*reward) for all time steps.
Gamma = 0 is present only.
Gamma = 1 is future only.

Nu = 1/Pr(o|b,a) is the normalizing constant with
Pr(o|b,a) = sum(O taken at s',a) * sum(T(s'|s,a)b(s))

belief state = b'
b'(s') = Nu*O(o|s',a) * sum(T(s'|s,a)b(s))
"""

class POMDP(object):
    #gamma is the discount factor, which discounts future rewards. Z is observations.
    #O is observation function - or state probability
    def __init__(self, states, actions, Z, O, terminals, gamma):
        self.states = [states]
    def T(state,action):
        return new_state

    def R(self,state,action):
        return reward

    def actions(self,state):
        return action

#probability that we are in state given our observation Z
    def O(state,Z):

"""
Belief MDP = (B,A,Tau,r,Gamma)
B = Belief state
A = actions
Tau = transition probabilities
r = reward
gamma = discount

Tau and r need to be derived from the original POMDP
Tau(b,a,b') = sum(Pr(b'|b,a,o)*Pr(o|b,a))
(Pr(b'|b,a,o) = {1} if the belief update returns b'. {0} otherwise

r is the expected reward from POMDP over the belief function
r(b,a) = sum(b(s)R(s,a))
"""

class BeliefMPD(object):
    def __init__(self,belief,actions,tau,r,gamma):

    def belief_update(self):
        return belief

def policy_update():
    return policy

def expected_utility():
    return utility
