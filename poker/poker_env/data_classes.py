import copy
import numpy as np
from collections import deque
from random import shuffle
from poker_env.datatypes import Globals,Action,SUITS,RANKS


def flatten(l):
    return [item for sublist in l for item in sublist]

CARDS = []
for i in range(RANKS.LOW,RANKS.HIGH):
    for j in range(SUITS.LOW,SUITS.HIGH):
        CARDS.append([i,j])

class Status(object):
    PADDING = 'padding'
    ACTIVE = 'active'
    FOLDED = 'folded'
    ALLIN = 'allin'

STATUS_DICT = {
    Status.PADDING:0,
    Status.ACTIVE:1,
    Status.ALLIN:2,
    Status.FOLDED:3
}

class PlayerIndex(object):
    def __init__(self,n_players:int,street:int):
        """
        Class for keeping track of whose turn it is.
        current_index: int
        starting_index: int
        starting_street: int
        n_players: int
        """
        self.n_players = n_players
        self.starting_street = street
        self.starting_index = Globals.STARTING_INDEX[n_players][street]
        self.current_index = self.starting_index
        assert isinstance(street,int)
        assert isinstance(self.current_index,int)
        assert isinstance(self.n_players,int)
        assert isinstance(self.starting_index,int)

    def increment(self):
        self.current_index = max((self.current_index + 1) % (self.n_players + Action.OFFSET),1)

    def reset(self):
        self.current_index = self.starting_index

    def next_street(self,street):
        self.current_index = Globals.STARTING_INDEX[self.n_players][street]

    def update(self,value):
        self.current_index = copy.copy(value)

    def value(self):
        return self.current_index

    def __eq__(self,other):
        return True if self.current_index == other else False

    def __ne__(self, other):
        return not(self.current_index == other)

    def __str__(self):
        return Globals.POSITION_INDEX[self.current_index]

    def __index__(self):
        return self.current_index

    def __repr__(self):
        return f' Starting index {self.starting_index}, Current index {self.current_index}, N players {self.n_players}, Starting street {self.starting_street}'

    def clone(self):
        temp = PlayerIndex(self.n_players,self.starting_street)
        temp.update(self.current_index)
        return temp

    @property
    def key(self):
        return Globals.POSITION_INDEX[self.current_index]

class Player(object):
    def __init__(self,position,stack,street_total=0,status=Status,hand=None):
        """Status: Active,Folded,Allin"""
        self.position = position
        self.stack = stack
        self.street_total = street_total
        self.status = Status.ACTIVE
        self.hand = hand
        self.handrank = None
    
    def update_hand(self,hand):
        self.hand = hand
        
class Players(object):
    def __init__(self,n_players,starting_stack,cards_per_player):
        """Status: Active,Folded,Allin"""
        self.n_players = n_players
        self.starting_stack = starting_stack
        self.cards_per_player = cards_per_player
        self.initial_positions = Globals.PLAYERS_POSITIONS_DICT[n_players]
        self.reset()

    def reset(self):
        self.players = {position:Player(position,copy.copy(self.starting_stack)) for position in self.initial_positions}
        for player in self.players.values():
            assert(player.status == Status.ACTIVE)
            assert(player.stack == self.starting_stack)
            assert(player.street_total == 0)
    
    def initialize_hands(self,hands):
        for i,position in enumerate(self.initial_positions):
            start = i*self.cards_per_player
            end = (i+1)*self.cards_per_player
            player_cards = hands[start:end]
            self.players[position].update_hand(player_cards)
    
    def update_stack(self,amount,position:str):
        """
        Updates player stack,street_total and status after putting money into the pot.
        """
        assert isinstance(position,str)
        self.players[position].stack += amount
        self.players[position].street_total -= amount
        if self.players[position].stack == 0:
            self.players[position].status = Status.ALLIN
        assert self.players[position].stack >= 0,'Player stack below zero'

    def reset_street_totals(self):
        for player in self.players.values():
            player.street_total = 0

    def update_status(self,player,status):
        self.players[player].status = status

    def info(self,player):
        """Returns position,stack,street_total for given player"""
        return [Globals.NAME_INDEX[self.players[player].position],self.players[player].stack,self.players[player].street_total,STATUS_DICT[self.players[player].status]]

    def hero_info(self,player):
        """Returns position,stack,hand for given player"""
        return [Globals.NAME_INDEX[self.players[player].position],self.players[player].stack,*flatten(self.players[player].hand)]

    def observation_info(self,hero):
        """Returns position,stack,hand for all players other than hero"""
        return flatten([[Globals.NAME_INDEX[self.players[player].position],self.players[player].stack,*flatten(self.players[player].hand)] for player in self.initial_positions if player != hero])

    def return_active_hands(self):
        hands = []
        positions = []
        for i,position in enumerate(self.initial_positions):
            if self.players[position].status != Status.FOLDED:
                hands.append(self.players[position].hand)
                positions.append(position)
        return hands,positions

    def __getitem__(self,key):
        return self.players[key]

    def __len__(self):
        return len(self.players.keys())

    @property
    def num_active_players(self):
        return [self.players[player].status for player in self.players if self.players[player].status == Status.ACTIVE].count(Status.ACTIVE)
    
    @property
    def num_folded_players(self):
        return [self.players[player].status for player in self.players if self.players[player].status == Status.FOLDED].count(Status.FOLDED)
    
    @property
    def to_showdown(self):
        """Fast forwards to showdown if all players are allin"""
        statuses = [self.players[position].status for position in self.initial_positions]
        if Status.ACTIVE not in statuses and statuses.count(Status.ALLIN) > 1:
            return True
        return False

class LastAggression(PlayerIndex):
    def __init__(self,n_players,street):
        self.n_players = n_players
        self.starting_street = street
        self.starting_index = Globals.STARTING_INDEX[n_players][street]
        self.current_index = self.starting_index
        self.starting_action,self.starting_betsize = Globals.STARTING_AGGRESSION[street]
        self.aggressive_action = self.starting_action
        self.aggressive_betsize = self.starting_betsize

    def player_values(self):
        return [self.current_index,self.action,self.betsize]

    def update_aggression(self,position,action,betsize):
        self.current_index = copy.copy(position)
        self.aggressive_action = action
        self.aggressive_betsize = betsize

    def next_street(self,street):
        self.current_index = Globals.STARTING_INDEX[self.n_players][street]
        self.aggressive_action = Action.UNOPENED
        self.aggressive_betsize = 0

    def reset(self):
        self.current_index = self.starting_index
        self.aggressive_action = self.starting_action
        self.aggressive_betsize = self.starting_betsize

    @property
    def action(self):
        return self.aggressive_action

    @property
    def betsize(self):
        return self.aggressive_betsize

class Deck(object):
    def __init__(self,preset):
        self.reset(preset)

    def reset(self,preset):
        if preset:
            self.deck = preset
        else:
            self.deck = deque(copy.deepcopy(CARDS),maxlen=52)
    
    def deal(self,N):
        """Returns a list of cards (rank,suit)"""
        cards = []
        for card in range(N):
            cards.append(self.deck.pop())
        return cards

    def initialize_board(self,street):
        assert isinstance(street,int),f'street type incorrect {type(street)}'
        num_cards = Globals.INITIALIZE_BOARD_CARDS[street]
        return self.deal(num_cards)

    def deal_board(self,street):
        assert isinstance(street,int),f'street type incorrect {type(street)}'
        num_cards = Globals.ADDITIONAL_BOARD_CARDS[street]
        return self.deal(num_cards)

    def shuffle(self):
        shuffle(self.deck)

    def __len__(self):
        return len(self.deck)

class GlobalState(object):
    def __init__(self,mapping):
        self.mapping = mapping
        self.reset()

    def add(self,state):
        self.global_states.append(state)

    def stack(self):
        return np.vstack(self.global_states)

    def reset(self):
        self.global_states = []

    @property
    def last_aggressive_action(self):
        return self.global_states[-1][self.mapping['last_aggressive_action']][0]
        
    @property
    def last_aggressive_betsize(self):
        return self.global_states[-1][self.mapping['last_aggressive_betsize']][0]

    @property
    def last_aggressive_position(self):
        return self.global_states[-1][self.mapping['last_aggressive_position']][0]
        
    @property
    def last_action(self):
        return self.global_states[-1][self.mapping['last_action']][0]
        
    @property
    def last_betsize(self):
        return self.global_states[-1][self.mapping['last_betsize']][0]

    @property
    def penultimate_betsize(self):
        if len(self.global_states) > 1:
            return self.global_states[-2][self.mapping['last_betsize']][0]
        return 0
