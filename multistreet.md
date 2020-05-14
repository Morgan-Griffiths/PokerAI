# Multistreet poker

One of the critical things about multistreet poker is that the state,observation is the full stack of all of the decision points made by all players up to that point. There are two obvious ways to support that. One is by selecting a trajectory maxlen and padding states and observations. The other is storing each decision point piecemeal.

## State construction

The state should be a series of decision points. Which when stacked together create a complete picture of what the current situation is. The components of what make up a poker game state: Hero hand, board, Hero position, Hero stacksize, Positions of currently active players (still holding cards) in order of action (plus positional embedding of positional ordering) and their stacksizes. The pot size. The last action, Last betsize, Last position who acted. We will also need to add an extra positional embedding for the dealer, for when there is no last action because its a new street.

In heads up we can cut some corners here because there are only two players. The last player who acted will always be our opponent. Unless we just entered a new street, in which case the last player will be the dealer.

Occasionally there will be a skip between decision points. Such as (In Heads up play) when BB closes the action preflop (street 0) and the board cards are dealt, and BB acts first.

We must record the last decision in the context in which BB made it. And then record the new cards and the fact that the game now is unopened because it is a new street. Those two states stacked, will be part of the input for BB, along with all the prior states. Additionally we must record the actions of other from the perspective of Hero. 

Conv Cards
Positional Embedding
Ordering Embedding (add to positional embedding for order)
Action embedding

P1 Hand,Board,street,P1 position,P2 position,Previous_action,Previous_betsize,P1 Stack,P2 Stack,Amnt to call,Pot odds
        

## Padding

Upside of padding is that you can train in batches (because all the states and observations have the same length). 

## Preflop

Preflop actions are a little wacky. Technically the posted blinds are blind bets and raises. However because they are blind, they still get an option if someone calls. Which turns the typical round over logic on its head. Because calling does not necessarily mean the end of the round.

Presumably a more generic solution will have to take into account the blind bet as a separate thing, with an option. Or record the fact that they haven't acted yet. In multi person poker, it matters about who still has the option to act, so calling does not mean the end of the round in multiway pots. And the action can be reopened if a later player raises.

So the unusual circumstance of SB calling, and BB getting the options of raise/check. Is solved by counting number of actor raises. If actor raises > 0 then calling will close the action and move on to the next street.

### Blind state construction

I created a special function to seed the state with the posted blinds first. and add to the history. The actions are recorded as bet and raise. And then SB is considered as facing a raise. Except for the fact that calling does not end the action. Due to this, the street parameter will be important for the network.

