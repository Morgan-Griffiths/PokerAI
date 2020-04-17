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

Because flushes are a binary outcome, we can calculate them separately and remove that axis from the card points. Therefore now we have each card on the x axis and ranks on the y axis. If we convolve a rank sorted hand, with a 5x5 kernel, it should be able to pick up all the hand categories. Because it becomes the detection of lines. 5x5 is just enough to get a full activation on straights. But if the hand is a partial straight, AKA high card. Then the strength of the activation will be less. Allow us to transform the data such that the final layer can separate each category cleanly.

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