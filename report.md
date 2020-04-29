# Solving Kuhn

I used a simple baseline of reinforce to solve simple and complex kuhn poker. The downside of this is that when i update the network, it updates the action proportionally to the reward. This is why you will see in the performance graphs, things like A checking rising and falling in frequency. Because by checking A a reward of 1 is guaranteed, which means checking will be increased whenever checking is selected. Betting has an expectation of greater than 1, and thus occasionally will be updated 2x as much as checking. But due to the randomness of action selection, there can be periods of time when only checking is selected, which results in the checking frequency increasing significantly. 

## Improvements

Clearly it would be preferred to have a sense of how good the state is, and then update towards how effect that action is relative to the others. In the case of SB having an A. The EV of betting is > 1. Whereas the EV of checking is == 1. therefore when selecting check, we want to move the frequency down even though the outcome is positive. A way around this is by introducing a critic, that estimates the value of a state. Or estimates the Q values (action,state pairs). And then update the action frequency towards the critic's valuation.

## Critic

A key aspect of the critic's ability to correctly determine the outcome is the ability to distinguish which actions took place previously and which action is the current one taken by the actor. There are several options to encoding the positional information. Create a positional embedding vector along with the embedding of the action and add the two. One hot encode the actions and use a convnet.