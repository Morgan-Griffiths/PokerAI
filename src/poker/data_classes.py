import pickle
import torch
import numpy as np
from collections import deque
from random import shuffle

from cardlib import winner,encode,decode,holdem_hand_rank,holdem_winner

action_dict = {0:'check',1:'bet',2:'call',3:'fold',4:'raise',5:'unopened'}
position_dict = {2:['SB','BB'],3:['SB','BB','BTN'],4:['SB','BB','CO','BTN'],5:['SB','BB','MP','CO','BTN'],6:['SB','BB','UTG','MP','CO','BTN']}

class Action(object):
    def __init__(self,action):
        self.action = action
        
    def item(self):
        return self.action
        
    def readable(self):
        return action_dict[self.action]
    
class Card(object):
    def __init__(self,rank,suit):
        self.rank = rank
        self.suit = suit
        
    def __str__(self):
        printable = f'Rank {self.rank}, Suit {self.suit}' if self.suit else f'Rank {self.rank}'
        return printable
    
class Deck(object):
    def __init__(self,ranks,suits):
        self.ranks = ranks
        self.suits = suits
        self.deck = self.construct_deck()
    
    def construct_deck(self):
        deck = deque(maxlen=52)
        if self.suits is None:
            for rank in self.ranks:
                deck.append(Card(rank,None))
        else:
            for rank in self.ranks:
                for suit in self.suits:
                    deck.append(Card(rank,suit))
        return deck
    
    def deal(self,N):
        cards = []
        for card in range(N):
            cards.append(self.deck.pop())
        return cards
    
    def shuffle(self):
        shuffle(self.deck)
        
    def reset(self):
        self.deck = self.construct_deck()
        
    def display(self):
        for card in self.deck:
            print(card)

class Historical_point(object):
    def __init__(self,player,action):
        self.player = player
        self.action = Action(action)
    
    def display(self):
        return (self.player,self.action.readable())
    
class History(object):
    def __init__(self,initial_state=None):
        if initial_state:
            self.history = initial_state
        else:
            self.history = []
        
    def add(self,player,action):
        self.history.append(Historical_point(player,action))
        
    def display(self):
        for step in self.history:
            print(step.display())
            
    @property
    def last_action(self):
        if len(self.history) > 0:
            return self.history[-1].action.item()
        return None

    @property
    def penultimate_action(self):
        if len(self.history) > 1:
            return self.history[-2].action.item()
        return None

    def __len__(self):
        return len(self.history)
    
    def reset(self):
        self.history = []
            
class GameTurn(object):
    def __init__(self,initial_value=0):
        self.initial_value = initial_value
        self.value = initial_value
        
    def increment(self):
        self.value += 1
        
    def reset(self):
        self.value = self.initial_value
    
'''
In the Future we will need to account for side pots
'''
class Pot(object):
    def __init__(self,initial_value:int):
        self.initial_value = initial_value
        self.value = initial_value
        
    def add(self,amount):
        self.value += amount
        
    def update(self,amount):
        self.initial_value = amount
        self.value = amount
        
    def reset(self):
        self.value = self.initial_value
        
class PlayerTurn(object):
    def __init__(self,initial_value:int,max_value=2,min_value=0):
        self.initial_value = initial_value
        self.value = initial_value
        self.min_value = min_value
        self.max_value = max_value
    
    def increment(self):
        self.value = max(((self.value + 1) % self.max_value),self.min_value)
        
    def reset(self):
        self.value = self.initial_value

class Player(object):
    def __init__(self,hand,stack,position,active=True):
        self.hand = hand
        self.stack = stack
        self.position = position
        self.active = active
        
class Players(object):
    def __init__(self,n_players:int,stacksizes:list,hands:list,to_act:str):
        self.n_players = n_players
        self.stacksizes = stacksizes
        self.to_act = to_act
        self.initial_positions = position_dict[n_players]
        self.reset(hands)

    def update_hands(self,hands):
        self.hands = hands
        for i,position in enumerate(self.initial_positions):
            self.players[position].hand = self.hands[i]
        
    def reset(self,hands:list):
        self.hands = hands
        self.poker_positions = deque(self.initial_positions,maxlen=9)
        self.players = {position:Player(hands[i],self.stacksizes[i].clone(),position) for i,position in enumerate(self.poker_positions)}
        self.game_states = {position:[] for position in self.poker_positions}
        self.observations = {position:[] for position in self.poker_positions}
        self.actions = {position:[] for position in self.poker_positions}
        self.action_probs = {position:[] for position in self.poker_positions}
        self.rewards = {position:[] for position in self.poker_positions}
        self.player_turns = {position:0 for position in self.poker_positions}
        
    def store_states(self,state:torch.Tensor,obs:torch.Tensor):
        self.observations[self.current_player].append(obs)
        self.game_states[self.current_player].append(state)
        
    def store_actions(self,action:int,action_probs:torch.Tensor):
        self.actions[self.current_player].append(action)
        self.action_probs[self.current_player].append(action_probs)
        
    def store_rewards(self,position:str,reward:float):
        N = self.player_turns[position]
        torch_rewards = torch.Tensor(N).fill_(reward)#.view(N,1)
        self.rewards[position].append(torch_rewards)
        
    def update_stack(self,amount:float,player=None):
        if player == None:
            self.players[self.current_player].stack += amount
        else:
            self.players[player].stack += amount
        
    def order_post_flop(self):
        pass
        
    def increment(self):
        self.player_turns[self.current_player] += 1
        self.poker_positions.rotate(1)
        
    def gen_rewards(self):
        for i,initial_stacksize in enumerate(self.stacksizes):
            position = position_dict[self.n_players][i]
            player = self.players[position]
#             print(f'gen_rewards,player.stack {player.stack}, initial_stacksize {initial_stacksize}')
            self.store_rewards(position,player.stack - initial_stacksize)
    
    def get_player(self,position):
        return self.players[position]
    
    
    def get_hands(self):
        '''
        Later will take an argument for active players
        '''
        hands = []
        for player in range(self.n_players):
            position = self.initial_positions[player]
            hands.append(self.players[position].hand)
        return hands
    
    def get_inputs(self):
        ml_inputs = {}
        del_positions = set()
        for position in self.initial_positions:
            if len(self.actions[position]):
                ml_inputs[position] = {
                    'game_states':self.game_states[position],
                    'observations':self.observations[position],
                    'actions':self.actions[position],
                    'action_probs':self.action_probs[position],
                    'rewards':self.rewards[position]
                }
                assert(self.rewards[position][0].size(0) == len(self.actions[position]))
            else:
                if position not in del_positions:
                    del_positions.add(position)
        for position in del_positions:
            if position in ml_inputs:
                del ml_inputs[position]
        return ml_inputs
    
    def get_stats(self):
        stats = {}
        for position in self.initial_positions:
            stats[position] = {
                'hand':self.players[position].hand,
                'position':position,
                'stack':self.players[position].stack,
            }
        return stats
        
    @property
    def current_hand(self):
        return self.players[self.current_player].hand
    
    @property
    def current_player(self):
        return self.poker_positions[0]
    
    @property
    def previous_player(self):
        return self.poker_positions[-1]
    
class Rules(object):
    def __init__(self,params,game):
        self.game = game
        if self.game == 'kuhn':
            self.load_rules(params)
        else:
            raise ValueError('Game type not supported')

    def return_mask(self,state):
        return self.mask_dict[state[0,self.action_index].long().item()]
            
    def load_rules(self,params):
        self.unopened_action = params['unopened_action']
        self.action_dict = params['action_dict']
        self.betsize_dict = params['betsize_dict']
        self.mask_dict = params['mask_dict']
        self.bets_per_street = params['bets_per_street']
        self.raises_per_street = params['raises_per_street']
        self.db_mapping = params['mapping']
        self.action_index = params['action_index'] # Indexes into game_state. Important for masking actions
        self.state_index = params['state_index']
        self.action_space = len(self.action_dict.keys())
        self.over = self.kuhn1street if self.bets_per_street == 1 else self.kuhn2street

    def kuhn2street(self,env):
        done = False
        if env.history.last_action == 3 or env.history.last_action == 2:
            done = True
        if len(env.history) > 1:
            if env.history.last_action == 0 and env.history.penultimate_action == 0:
                done = True
        return done
        
    def kuhn1street(self,env):
        done = False
        if env.history.last_action == 3 or env.history.last_action == 2 or env.history.last_action == 0:
            done = True
        return done
    
def eval_kuhn(hands):
    ranks = [card.rank for card in hands]
    return np.argmax(ranks)

def eval_holdem(hands):
    hand1,hand2,board = hands
    hand1 = [[card.rank,card.suit] for card in hand1]
    hand2 = [[card.rank,card.suit] for card in hand2]
    board = [[card.rank,card.suit] for card in board]
    en_hand1 = [encode(c) for c in hand1]
    en_hand2 = [encode(c) for c in hand2]
    en_board = [encode(c) for c in board]
    return holdem_winner(en_hand1,en_hand2,en_board)

def eval_omaha_hi(hands):
    hand1,hand2,board = hands
    hand1 = [[card.rank,card.suit] for card in hand1]
    hand2 = [[card.rank,card.suit] for card in hand2]
    board = [[card.rank,card.suit] for card in board]
    en_hand1 = [encode(c) for c in hand1]
    en_hand2 = [encode(c) for c in hand2]
    en_board = [encode(c) for c in board]
    return  winner(en_hand1,en_hand2,en_board)

class Evaluator(object):
    def __init__(self,game):
        self.game = game
        if self.game == 'kuhn':
            self.evaluate = eval_kuhn
        elif self.game == 'holdem':
            self.evaluate = eval_holdem
        elif self.game == 'omahaHI':
            self.evaluate = eval_omaha_hi
        else:
            raise ValueError('Game type not supported')
        
    def __call__(self,hands):
        return self.evaluate(hands)