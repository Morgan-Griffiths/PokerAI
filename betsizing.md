# Betsizing

In addition to outputting an action, if the action is bet or raise, we must also choose a bet size.

There are multiple ways to go about implementing this. Initially for the actor network, i was going to output a single vector of action categories. Which contains both the action categories like Fold, Raise, Check... but also contains the subcategories of bet 50% of pot, 100% of pot etc. 

An issue with this approach, is that each bet size is treated separately, initially the probabilities will be over all action categories and sub categories in a flat structure. This means initially betting will be by far the most probable action, and in order to reduce betting as a whole, it would be necessary to increase the probability of one of the other main action categories like folding/checking. As opposed to decreasing unsuccessful bets, which will increase the probability of other bets, leading to a wackamole situation, especially if the number of betsizes is large. 

A potential solution to this is to break the network into two sections, the first section computes a probability over categories and picks one. If the category is bet or raise and requires a betsize, then it continues into the next section which also outputs a betsize. Otherwise the betsize is 0. This seems like it should improve learning because it is easier for it to make a choice at the category level, instead of a blend of category and subcategory. 

A nice feature of this style of network is that the outputs are readily recognizable. An annoyance is the fact that we will have to check for none zero bets to perform betsize gradient updates which makes the learning process more complicated. And the network itself is more complicated. 

It also opens up the possibility of using a regressive output that avoids the descretization of betsizes. 