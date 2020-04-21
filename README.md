# PokerAI

A combination of poker environment simulator and a bitwise Omaha hand winner evaluator written in Rust. 

## Requirements

There are two requirement files, one for pip and one for conda.
```pip install pip_requirements.txt```
or 
```conda create --name <env> --file conda_requirements.txt```

## To build the Omaha Evaluator

to build rust code, cd into rusteval and run

```
cargo build --release
```
(`brew install rust` if you don't have it.)

# Abstract

A series of poker environments that cover each of the individual complexities of poker, allowing one to test networks and learning architectures quickly and easily starting from the simpliest env all the way to real world poker. The goal is a single API that interfaces with the training architecture for the agent such that you can scale the complexity as needed. Asserting that the learning algorithm learns at all stages.

# Using the library

Build the data and all the folders by ```python setup.py```

Build a specific dataset with ```python build_dataset.py -d <dataset>```

Modify poker/models/network_config.py to change which network to train. Add or modify poker/models/networks.py to try different models.

Train a network (loaded from network_config) on a dataset with ```python cards.py -d <dataset> -M train```
Examine a network's output (loaded from network_config) on a dataset with ```python cards.py -d <dataset> -M examine```

# Poker Environments

There are a number of environments, each increasing in complexity.

# Kuhn

- Deck [A,K,Q]
- Betsize fixed at 1
- Raise size fixed at 2
- Initial pot of 1
- 1 Street

## Simple Kuhn

*SB Options:*
- Check,bet,fold

*BB Options:* _facing bet only_
- Call,fold

#### Solution:

*SB*
- Q should mostly fold/check (equal actions). bet occasionally
- K should check entirely
- A should bet entirely

_Baseline performance_

![Graph](Action_probabilities_for_SB.png)

*BB*
- Q fold always
- K facing bet, should call occasionally
- A call always

_Baseline performance_

![Graph](Action_probabilities_for_BB.png)

## Complex Kuhn

*SB Options:*
- Check,Bet
- Call,Fold facing raise

*BB Options:*
- Bet,Check facing Check
- Call,Raise,Fold facing Bet

#### Solution:

*SB*
- Q should mostly fold/check (equal actions). bet occasionally
- K should check entirely
- A should bet entirely

_Baseline performance_

![Graph](Complex_Action_probabilities_for_SB.png)

*BB*
- Q fold always
- K facing bet, should call occasionally
- A Raise always

_Baseline performance_

![Graph](Complex_Action_probabilities_for_BB.png)

# Decoding cards

In omaha each hand has 4 cards. The final board has 5. Each player must use two cards from their hand and 3 from the board. Because this is potentially a difficult operation for NNs, initially the process is simplified by only predicting handtype given 5 cards. Minimally the network must be capable of distinguishing all the relevant handtypes in hand+board combinations. Ideally can distinguish hand strengths within each handtype, which can be measured by predicting the winner between two hands of the same handtype. 

In the scenario of 4 hand cards and 5 board cards. We have 60 combos of 5 card hands. we can use the same methods for solving the 5 card hands, and potentially take the max of the outputs to determine the best hand.

In real play, we will need to know how our hand interfaces with the board. And which suits we have with which cards. This not only will let us know how big our flush is, but also if we have some relevant flush blockers, when we don't have a flush.

We will also potentially want to make preflop decisions based on our hand alone. This could mean we want an additional embedding of the hand to inform those decisions.

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

### How to deal with Hole Cards with varying amounts of information about the board

- Could pad the board and convolve with partial information. Need to pass the street as a parameter?
- Potentially the network already knows enough to determine which street it is.
- Could embed the street and add it to the outputs.
- Could add the street as another axis? 
- Could embed the hand and also convolve the hand board.

## Added betsize

The important part about betsizing is that if i break actions into categories and then within those categories you can choose sizing. Improper sizing will result in the category not being chosen as often. Conversely, if i use a critic, the critic must be able to take an action an a betsize. Ideally you update both against the betsize and the action not just the action category. Additionally its important to be able to have mixed strategies. So either gaussian or descrete categorical output for betsize is also preferred. such that different categories can be reinforced. 

Additional levels to network that outputs a analog value, which is a % of pot. 

Will test initially two sizes 0.5p and 1p along with check,fold etc. All as a categorical output with the action space descretized. Then scale up to something like 100 descretized.

## Multiple streets

Dealing with histories. Record only actions and game situations? or include board and hands.
- Transformer
- LSTM

## Full game

Possibilities:
MuZero-esque. Dynamics Model (samples outcomes), Villain model (predicts opponents actions), Predicting next card.
- Transformer/LSTM
- Hidden dynamics with optional recursion.

