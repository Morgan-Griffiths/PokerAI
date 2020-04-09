# PokerAI

A combination of poker environment simulator and a bitwise Omaha hand winner evaluator written in Rust. 

## To build the Omaha Evaluator

to build rust code, cd into rusteval and run

```
cargo build --release
```
(you should have rust installed ofc. `brew install rust` if you don't have it yet.)

## Source code is contained within src

# Poker Environments

There are a number of environments, each increasing in complexity.

# Kuhn

- Deck [A,K,Q]
- Betsize fixed at 1
- Raise size fixed at 2
- Initial pot of 1
- 1 Street

## Simple Kuhn

SB Options:
Check,bet,fold

BB Options: *facing bet only*
Call,fold

Solution:

SB 
Q should mostly fold/check (equal actions). bet occasionally
K should check entirely
A should bet entirely

BB
Q fold always
K facing bet, should call occasionally
A call always

## Complex Kuhn

SB Options:
Check,Bet
Call,Fold facing raise

BB Options:
Bet,Check facing Check
Call,Raise,Fold facing Bet

Solution:

SB 
Q should mostly fold/check (equal actions). bet occasionally
K should check entirely
A should bet entirely

BB
Q fold always
K facing bet, should call occasionally
A Raise always

## Decoding cards

- Predicting hand outcomes given two hands and a board.
- Predicting hand outcomes given partial or no board and two hands
- Predicting outcomes given hand and board

### Networks

- Multiheaded attention?
- Convnets

## Added betsize

## Multiple streets

## Full game