# PokerAI

A combination of poker environment simulator and a bitwise Omaha hand winner evaluator written in Rust. 

## To build the Omaha Evaluator

to build rust code, cd into rusteval and run

```
cargo build --release
```
(you should have rust installed ofc. `brew install rust` if you don't have it yet.)

# Abstract

A series of poker environments that cover each of the individual complexities of poker, allowing one to test networks and learning architectures quickly and easily starting from the simpliest env all the way to real world poker. The goal is a single API that interfaces with the training architecture for the agent such that you can scale the complexity as needed. Asserting that the learning algorithm learns at all stages.

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

## Decoding cards

Tests: 
- 13 cards in a row
- try with the board obscured.
- 9 cards without villain hand
- 4 cards without board and villain.
- 60 combinations of 2 cards of heros hand and 3 of board + the same for villain
- Hero combinations with mystery board cards

Predicting hand type:
- high card
- pair
- two pair
- 3 of a kind
- straight
- flush
- FH
- quads
- straight flush

Final test group has all hand strengths

### Networks

Layer that takes 5 cards and outputs a strength signal?
process the hand, process the hand + board
process hand, process board, add together

- Convnet + Multiheaded attention? Attention could select a particular hand of interest
- Convnet + FC
- Embedding + FC

### How to deal with Hole Cards with varying amounts of information about the board

- Mask inputs
- Run convolutions/fc on hidden state. Encode cards first and then pass into fc.

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

