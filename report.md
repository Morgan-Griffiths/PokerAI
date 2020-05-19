# Solving Kuhn

I used a simple baseline of reinforce to solve simple and complex kuhn poker. The downside of this is that when i update the network, it updates the action proportionally to the reward. This is why you will see in the performance graphs, things like A checking rising and falling in frequency. Because by checking A a reward of 1 is guaranteed, which means checking will be increased whenever checking is selected. Betting has an expectation of greater than 1, and thus occasionally will be updated 2x as much as checking. But due to the randomness of action selection, there can be periods of time when only checking is selected, which results in the checking frequency increasing significantly. 

## Improvements

Clearly it would be preferred to have a sense of how good the state is, and then update towards how effect that action is relative to the others. In the case of SB having an A. The EV of betting is > 1. Whereas the EV of checking is == 1. therefore when selecting check, we want to move the frequency down even though the outcome is positive. A way around this is by introducing a critic, that estimates the value of a state. Or estimates the Q values (action,state pairs). And then update the action frequency towards the critic's valuation.

Further improvements can be in training the policy based on an advantage function that rewards/punishes relative to optimal and none optimal actions. One tricky thing about this, is it relies on having a value function that accurately predicts values across actions. When the values are not accurate, it can lead to local optima and crash learning. Due to this i added an exploration noise layer in the actor, to create some noise to the actors actions and explore more of the game, thus giving the critic a more detailed value function.

Ultimately, mixed strategies are the optimal result for poker. This requires that the value functions be the same, yet the frequencies of the actions be different. As the value of the action corresponds to its frequency.

## Critic

A key aspect of the critic's ability to correctly determine the outcome is the ability to distinguish which actions took place previously and which action is the current one taken by the actor. There are several options to encoding the positional information. Create a positional embedding vector along with the embedding of the action and add the two. One hot encode the actions and use a convnet.

The positional embedding seems to work best. Additionally, with the advent of changing the policy update function to **V(s,a) - E(s)** required changing the critic from Q(s,a) to Q(s) -> outputting values for each action A. Then selecting the corresponding action value and updating the action probability. Updating the value Q(s,a) by minimizing the distance Q(s,a) = R_a - Q(s,a)

## The Unreliable Omniscient Critic

When it comes to imperfect information games, you would think more information is always preferred. I have read many papers detailing such actor critic methods with an omniscient critic who has perfect information informing an actor's actions. Such as MADDPG, AlphaStar, to name a few. However i don't recall any of them mentioning that in fact omniscience can hinder learning substantially. Part of this i'm sure comes down to information theory and what is recoverable by the actor who can only see the partial state. But instead of being entirely vague about this, i will show a concrete example where it encourages non optimal play.

Enter Kuhn poker. Kuhn poker has a deck of 3 cards A,K,Q. As you might expect, A > K > Q. If you hold an A, your opponent holds either K or Q with equal frequency. Player's take turns acting. The actions are the usual poker actions: Check,Bet,Fold,Raise,Call. Betsize is fixed at 1 (doesn't matter but makes the math simpler for our purposes). The current pot is 1. Each player has a stacksize of 2 Player 1 acts first. 

Suppose we are player 1 and we hold a Q. Our opponent holds an A or K. If we bet 1 into 1, intuitively we can see that we need our bluff to work 50% of the time. 50% of the time our opponent will have an A and raise (effectively calling here), so its up to the remaining times Player 2 has a K that decide if we can bluff here or not. To call with a king only requires 33% equity. But if he calls with any fruequency then betting with a Q is negative expected value or -EV. So any fraction of calling kings above 0 makes betting a Q not profitable. This actually means that its better to reduce the betsize from 1.

The optimal solution for Player 1 generally speaking, is to bet some fraction of Qs, all As. Check K and call or fold relative to our opponent's ratio of As to Qs. 

Player 2 will also bet/raise A, bet some fraction of Q, check/call K. 

From Player 2's perspective, if we hold a K and are facing a bet, Raising makes no sense as Player 1 will call A and fold Q. Effectively only costing us more money. But if we have an omniscient critic. When we hold K and are facing a Q, raising will always be of slightly higher value because of the times when our opponent called with a Q. When we hold a K and are facing an A, it will say fold. So we will be updated alternating between increasing raise and increasing fold. But the Actor has no way of telling what hand he is facing.

Whereas if the critic is not omniscient, the values of both situations will be combined, and it will be clear that the value of calling should be around 0 if the opponent is balancing his range. Folding is 0, and Raising is negative. Which means the actor will be correctly updated towards balancing between folding and calling.

I hope this simple example demonstrates how naively using more information in a critic in an imperfect information game can actually impede performance. It can still be used, but it would have to be used to update a model of the opponent's actions. Which could then be used to inform your own.