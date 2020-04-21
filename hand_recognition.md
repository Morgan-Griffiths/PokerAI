# Abstract

## Categorizing Poker Hands

Poker hands come in 8 general flavors.

- high card
- pair
- two pair
- 3 of a kind
- straight
- flush
- FH
- quads
- straight flush

If we consider a 3 dimensional space, x axis suits, y axis ranks, z axis cards. When considering a 5 card hand, we will have 5 (x,y) pairs. Assume the cards are sorted by rank. Now in order to determine whether there is a flush, the set of suits must have 1 entry, as they all must be of the same kind. There is a binary flush, not flush outcome. No pair would be disconnected ranks. Pair would be two dots on the same horizontal line. quads would be 4. A straight will be a diagonal line. If we can detect these seperate cases, then the network should be able to categorize the hands. 

Because flushes are a binary outcome, we can calculate them separately and remove that axis from the card points. Therefore now we have each card on the x axis and ranks on the y axis. If we convolve a rank sorted hand, with a 5x5 kernel, it should be able to pick up all the hand categories. Because it becomes the detection of lines. 5x5 is just enough to get a full activation on straights. But if the hand is a partial straight, AKA high card. Then the strength of the activation will be less. Allowing us to transform the data such that the final layer can separate each category cleanly.

In the real game, you will see 4 cards of your hand, and 5 cards on the board. This combines into 60 5 card combinations. To attain this underlying reality, you could hard code the combinatoric split into the network. However convnets have the ability to create all the combinations themselves. So it is sufficient to convolve the 9 cards with a 9x5 for ranks and 9x1 for suits.

# Decoding cards

In omaha each hand has 4 cards. The final board has 5. Each player must use two cards from their hand and 3 from the board. Because this is potentially a difficult operation for NNs, initially the process is simplified by only predicting handtype given 5 cards. Minimally the network must be capable of distinguishing all the relevant handtypes in hand+board combinations. Ideally can distinguish hand strengths within each handtype, which can be measured by predicting the winner between two hands of the same handtype. 

In the scenario of 4 hand cards and 5 board cards. We have 60 combos of 5 card hands. we can use the same methods for solving the 5 card hands, and potentially take the max of the outputs to determine the best hand.

In real play, we will need to know how our hand interfaces with the board. And which suits we have with which cards. This not only will let us know how big our flush is, but also if we have some relevant flush blockers, when we don't have a flush.

We will also potentially want to make preflop decisions based on our hand alone. This could mean we want an additional embedding of the hand to inform those decisions.

## Initial results

### First network

*input not sorted*

- Input (b,5,2)
- split suits and ranks
- 1 hot encode ranks
- convolve ranks with (5,5) 1d convnet
- MLP ranks with 3 FC
- suits Linear(5,1)
- combine with suits
- dropout

last layer dropout was a poor choice especially because the flush node was only one and without it the network would have no idea what the flush outcome was.

This resulted in 99.9%+ confidance in quads/FH/trips/twopair/pair. But was generally split 45/55 on straightflush/straight and flush/highcard. 

### Second network

*input not sorted*

- Input (b,5,2)
- split suits and ranks
- 1 hot encode ranks and suits
- convolve ranks with (5,5) 1d convnet
- convolve suits with (5,1) 1d convnet
- MLP ranks with 3 FC
- suits Linear(5,12)
- combine suits,ranks

trained on the test set for speed of results.
This resulted in 80-90%+ confidance in most categories. Occasionally confused straights with near straights. Very confidant on straight flushes. 

## Data Representation

### Predicting hand type:

- high card
- pair
- two pair
- 3 of a kind
- straight
- flush
- FH
- quads
- straight flush

#### Predicting rank handtypes given 5 cards

This can be solved by convolving a 5x5 kernel across the rank,card space.
*possibly other methods*

#### Predicting suit handtypes given 5 cards

FC didn't seem to work well.
This can be solved by convolving a 5x1 kernel across the suit,card space.
*possibly other methods*

#### Predicting handtype given 5 cards

Convolve rank and suit seperately, process rank in MLP and combine suit,rank output into final categorization layer.

#### Predicting handtype given 10 cards (two 5 card hands), predict winner - all handtypes

Comparing hands within classes.
- The target is -1,0 or 1 (2nd player wins, tie, first player wins)

Same architecture as with 5 cards works well. Can just convolve over both hands simultaneously

#### Predicting handtype given 9 cards
Same as above except:
- Input will be (b,9,2)
- Optionally the input can be (b,60,5,2) by changing "five_card_conversion" to true

Can clearly be solved if i change the representation to (60,5,2) shape.
Can it be solved without encoding the 60,5?

#### Predicting the winner given 13 cards
- Each rank is represented by a integer from 2-14 (2-A)
- Each suit is represented by a integer from 1-4 (s,h,c,d)
- A hand is (4,2)
- two hands and the board horizontally stacked is (13,2)
- Input will be (b,13,2)
- The target is -1,0 or 1 (2nd player wins, tie, first player wins)

Can clearly be solved if i change the representation to (60,10,2) shape.
Can it be solved without encoding the 60,10?

#### Blocker dataset 9 cards

Learning to distinguish individual cards in your hand and how they relate to the board

- board is always flush
- hand never has flush
- hand always has A
- hand either has A blocker or not
- target is 0,1 binary outcome - blocker not blocker

#### Partial boards

Learning representations that work for preflop/flop/turn/river

- Each building pass builds 8 different inputs. Hand only, Hand + flop, Hand + turn, Hand + river. for hero and villain
- target is -1,0,1

#### Handranks

- 5 card hand
- target is [0,7462]

### How to deal with Hole Cards with varying amounts of information about the board

- Could pad the board and convolve with partial information. Need to pass the street as a parameter?
- Potentially the network already knows enough to determine which street it is.
- Could embed the street and add it to the outputs.
- Could add the street as another axis? 
- Could embed the hand and also convolve the hand board.

## Generic solution

- C can be 5,9,10
- 13 cards must be broken down into two 9 card segments.
- convolve ranks with (C,5) 1d convnet
- convolve suits with (C,1) 1d convnet

