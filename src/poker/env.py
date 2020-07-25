from poker.spaces import Array,BoundedArray
import copy
import numpy as np
from random import shuffle
import poker.datatypes as pdt
from collections import deque
from utils.cardlib import hand_rank,encode

"""
last_position: int list of positions. last position is the null position. Used for the beginning of streets.
"""

ACTION_MASKS = {
        0:np.array([1,0,0,1,0]),
        1:np.array([0,0,0,0,0]),
        2:np.array([1,0,0,0,1]),
        3:np.array([0,1,1,0,1]),
        4:np.array([0,1,1,0,1]),
        5:np.array([1,0,0,1,0])
        }

BOARD_UPDATE = {
    1:(0,6),
    2:(6,8),
    3:(8,10)
}
POSITION_INDEX = {
    0:'SB',
    1:'BB',
    2:'BTN'
}
STARTING_INDEX = { 
    2:{
        0:0,
        1:1,
        2:1,
        3:1
    },
    3: {
        0:0,
        1:0,
        2:0,
        3:0
    }
}
NAME_INDEX = {v:k for k,v in POSITION_INDEX.items()}

STARTING_AGGRESSION = {
    0:(4,1),
    1:(5,0),
    2:(5,0),
    3:(5,0)
}

class Status(object):
    ACTIVE = 'active'
    FOLDED = 'folded'
    ALLIN = 'allin'

STATUS_DICT = {
    Status.ACTIVE:0,
    Status.ALLIN:1,
    Status.FOLDED:2
}

def flatten(l):
    return [item for sublist in l for item in sublist]

class PlayerIndex(object):
    def __init__(self,n_players,street):
        self.n_players = n_players
        self.starting_street = street
        self.starting_index = STARTING_INDEX[n_players][street]
        self.current_index = self.starting_index

    def increment(self):
        self.current_index = (self.current_index + 1) % self.n_players

    def reset(self):
        self.current_index = self.starting_index

    def next_street(self,street):
        self.current_index = STARTING_INDEX[self.n_players][street]

    def update(self,value):
        self.current_index = copy.copy(value)

    def value(self):
        return self.current_index

    def __eq__(self,other):
        return True if self.current_index == other else False

    def __ne__(self, other):
        return not(self.current_index == other)

    def __str__(self):
        return POSITION_INDEX[self.current_index]

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
        return POSITION_INDEX[self.current_index]

class Player(object):
    def __init__(self,position,stack,street_total=0,status=Status,hand=None):
        """Status: Active,Folded,Allin"""
        self.position = position
        self.stack = stack
        self.street_total = street_total
        self.status = Status.ACTIVE
        self.hand = hand
    
    def update_hand(self,hand):
        self.hand = hand
        
class Players(object):
    def __init__(self,n_players,starting_stack,cards_per_player):
        """Status: Active,Folded,Allin"""
        self.n_players = n_players
        self.starting_stack = starting_stack
        self.cards_per_player = cards_per_player
        self.initial_positions = pdt.Globals.PLAYERS_POSITIONS_DICT[n_players]
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
    
    def update_stack(self,amount:int,position:str):
        """
        Updates player stack,street_total and status after putting money into the pot.
        """
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
        return [NAME_INDEX[self.players[player].position],self.players[player].stack,self.players[player].street_total,STATUS_DICT[self.players[player].status]]

    def hero_info(self,player):
        """Returns position,stack,hand for given player"""
        return [NAME_INDEX[self.players[player].position],self.players[player].stack,*flatten(self.players[player].hand)]

    def observation_info(self,hero):
        """Returns position,stack,hand for all players other than hero"""
        return flatten([[NAME_INDEX[self.players[player].position],self.players[player].stack,*flatten(self.players[player].hand)] for player in self.initial_positions if player != hero])

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
        self.starting_index = STARTING_INDEX[n_players][street]
        self.current_index = self.starting_index
        self.starting_action,self.starting_betsize = STARTING_AGGRESSION[street]
        self.aggressive_action = self.starting_action
        self.aggressive_betsize = self.starting_betsize

    def player_values(self):
        return [self.current_index,self.action,self.betsize]

    def update_aggression(self,position,action,betsize):
        self.current_index = copy.copy(position)
        self.aggressive_action = action
        self.aggressive_betsize = betsize

    def next_street(self,street):
        self.current_index = self.n_players
        self.aggressive_action = 5
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
    def __init__(self):
        self.reset()

    def reset(self):
        self.deck = deque(maxlen=52)
        for i in range(2,15):
            for j in range(0,4):
                self.deck.append([i,j])

    def deal(self,N):
        """Returns a list of cards (rank,suit)"""
        cards = []
        for card in range(N):
            cards.append(self.deck.pop())
        return cards

    def initialize_board(self,street):
        assert isinstance(street,int),f'street type incorrect {type(street)}'
        num_cards = pdt.Globals.INITIALIZE_BOARD_CARDS[street]
        return self.deal(num_cards)

    def deal_board(self,street):
        assert isinstance(street,int),f'street type incorrect {type(street)}'
        num_cards = pdt.Globals.ADDITIONAL_BOARD_CARDS[street]
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

class Poker(object):
    def __init__(self,params):
        self.game = params['game']
        self.to_shuffle = params['shuffle']
        self.global_mapping = params['global_mapping']
        self.state_mapping = params['state_mapping']
        self.obs_mapping = params['obs_mapping']
        self.betsizes = params['betsizes']
        self.num_betsizes = len(self.betsizes)
        self.starting_street  = params['starting_street']
        self.street = self.starting_street
        self.starting_pot = params['pot']
        self.pot = params['pot']
        self.n_players = params['n_players']
        self.cards_per_player = params['cards_per_player']
        self.bet_type = params['bet_type']
        self.starting_stack = params['stacksize']
        self.global_states = GlobalState(params['global_mapping'])
        self.players = Players(self.n_players,self.starting_stack,self.cards_per_player)
        self.current_index = PlayerIndex(self.n_players,self.starting_street)
        self.last_aggressor = LastAggression(self.n_players,self.starting_street)
        self.players_remaining = self.n_players
        self.mask_dict = ACTION_MASKS
        betsize_funcs = {
            pdt.LimitTypes.LIMIT : self.return_limit_betsize,
            pdt.LimitTypes.NO_LIMIT : self.return_nolimit_betsize,
            pdt.LimitTypes.POT_LIMIT : self.return_potlimit_betsize,
        }
        self.return_betsize = betsize_funcs[self.bet_type]

        assert(self.n_players >= 2)
        assert(self.starting_stack >= 1)
        assert(self.cards_per_player >= 2)

    def initialize_board(self):
        self.board = [0] * 10
        starting_board_cards = flatten(self.deck.initialize_board(self.starting_street))
        self.board[:len(starting_board_cards)] = starting_board_cards
        
    def update_board(self):
        assert(self.street > 0 and self.street < 4)
        new_board_cards = flatten(self.deck.deal_board(self.street))
        start,end = BOARD_UPDATE[self.street]
        self.board[start:end] = new_board_cards
        
    def reset(self):
        self.global_states.reset()
        self.current_index.reset()
        self.last_aggressor.reset()
        self.players_remaining = self.n_players
        self.deck = Deck()
        if self.to_shuffle:
            self.deck.shuffle()
        self.street = self.starting_street
        # Pot
        self.pot = self.starting_pot
        # Board
        self.initialize_board()
        # Players
        self.players.reset()
        # Hands
        hands = self.deck.deal(self.cards_per_player*self.n_players)
        self.players.initialize_hands(hands)
        # Blinds
        if self.starting_street == 0:
            self.instantiate_blinds()
        else:
            self.store_global_state(last_position=self.n_players,last_action=5,last_betsize=0,blind=0)
        # Generate state and masks
        state,obs = self.return_state()
        action_mask,betsize_mask = self.return_masks(state)
        return state,obs,self.game_over(),action_mask,betsize_mask
        
    def instantiate_blinds(self):
        """Passed predetermined values to update state, until the desired state is reached"""
        SB_post = 0.5
        SB_action = pdt.Globals.REVERSE_ACTION_ORDER[pdt.Actions.BET]
        self.update_state(SB_action,SB_post,blind=1)
        BB_post = 1.
        BB_action = pdt.Globals.REVERSE_ACTION_ORDER[pdt.Actions.RAISE]
        self.update_state(BB_action,BB_post,blind=1)
    
    def step(self,inputs):
        """
        Increments the env state with current action,betsize.
        inputs : dict of ints
        """
        assert isinstance(inputs['action_category'],int)
        assert isinstance(inputs['betsize'],int)
        action = inputs['action_category']
        betsize = self.return_betsize(action,inputs['betsize'])
        self.update_state(action,betsize)
        if self.round_over():
            self.increment_street()
        done = self.game_over()
        if done:
            self.resolve_outcome()
        state,obs = self.return_state()
        action_mask,betsize_mask = self.return_masks(state)
        return state,obs,done,action_mask,betsize_mask

    def return_player_order(self):
        """Returns sequence of player data (position,stacksize,street_total"""
        player_data = []
        temp_index = self.current_index.clone()
        for i in range(self.n_players):
            position = POSITION_INDEX[temp_index.current_index]
            player_data.append(self.players.info(position))
            temp_index.increment()
        return flatten(player_data)
    
    def update_state(self,action,betsize,blind=0):
        """Updates the global state. Appends the new global state to storage"""
        if not blind:
            self.players_remaining -= 1
        if (action == pdt.Globals.REVERSE_ACTION_ORDER[pdt.Actions.RAISE] or action == pdt.Globals.REVERSE_ACTION_ORDER[pdt.Actions.BET]):
            self.last_aggressor.update_aggression(self.current_index.value(),action,betsize)
            if not blind:
                self.players_remaining = self.players.num_active_players - 1 # Current active player won't act again unless action is reopened
        elif action == pdt.Globals.REVERSE_ACTION_ORDER[pdt.Actions.FOLD]:
            self.players.update_status(self.current_player,Status.FOLDED)
        self.pot += betsize
        self.players.update_stack(-betsize,self.current_player)
        last_position = NAME_INDEX[self.current_player]
        self.increment_current_player_index()
        self.store_global_state(last_position,action,betsize,blind)
    
    def store_global_state(self,last_position,last_action,last_betsize,blind):
        """
        Returns the global state
        self.board,self.street,last_position,last_action,last_betsize,blind,to_call,pot_odds
        """
        # last aggression
        aggressor_values = self.last_aggressor.player_values()
        if self.last_aggressor.current_index == self.n_players:
            total_bet = 0
        else:
            total_bet = self.players[self.last_aggressor.key].street_total
        to_call = total_bet - self.players[self.current_player].street_total
        if to_call == 0:
            pot_odds = 0
        else:
            pot_odds = to_call / (self.pot + to_call)
        initial_data = [*self.board,self.street,*aggressor_values,last_position,last_action,last_betsize,blind,self.pot,to_call,pot_odds]
        player_data = self.return_player_order()
        global_state = np.array(initial_data+player_data)
        assert global_state.shape[-1] == self.global_space
        self.global_states.add(global_state)
    
    def increment_street(self):
        self.street += 1
        assert not self.street > pdt.Globals.REVERSE_STREET_DICT[pdt.Street.RIVER],'Street is greater than river'
        # clear previous street totals
        self.players.reset_street_totals()
        # update board
        self.update_board()
        # update current index and last aggression
        self.players_remaining = self.players.num_active_players
        self.street_starting_index()
        self.last_aggressor.next_street(self.street)
        self.store_global_state(last_position=self.n_players,last_action=5,last_betsize=0,blind=0)
        # Fast forward to river if allin
        if self.players.to_showdown:
            for _ in range(pdt.Globals.REVERSE_STREET_DICT[pdt.Street.RIVER] - self.street):
                self.street += 1
                self.update_board()
            self.store_global_state(last_position=self.n_players,last_action=5,last_betsize=0,blind=0)

    def street_starting_index(self):
        self.current_index.next_street(self.street)
        if self.players[self.current_index.key].status != Status.ACTIVE:
            for i in range(self.n_players):
                self.current_index.increment()
                if self.players[self.current_index.key].status == Status.ACTIVE:
                    break
    
    def increment_current_player_index(self):
        self.current_index.increment()
        for i in range(self.n_players):
            if self.players[self.current_index.key].status == Status.ACTIVE:
                break
            self.current_index.increment()

    def return_state(self):
        # add hero hand, position, stack, street_total
        states = self.global_states.stack()
        N = states.shape[0]
        # local state
        player_data = np.array(self.players.hero_info(self.current_player))
        player_data = np.tile(player_data,(N,1)) if N > 1 else player_data[None,]
        state = np.hstack((player_data,states))
        # obs
        player_obs = np.array(self.players.observation_info(self.current_player))
        player_obs = np.tile(player_obs,(N,1)) if N > 1 else player_obs[None,]
        obs = np.hstack((player_data,player_obs,states))
        assert state.shape[-1] == self.state_space
        assert obs.shape[-1] == self.observation_space
        return state[None,],obs[None,]
            
    def active_players(self):
        """Returns a list of active players"""
        actives = []
        active_index = self.current_index.clone()
        for i in range(self.n_players):
            if self.players[active_index].status == Status.ACTIVE:
                actives.append(active_index)
            active_index.increment()
        return actives
    
    def round_over(self):
        if self.players_remaining == 0 and self.street != 3 and self.players.num_folded_players != self.n_players - 1:
            return True
        return False
    
    def game_over(self):
        if (self.players_remaining == 0 and self.street == 3) or (self.players.num_folded_players == self.n_players - 1):
            return True
        return False
    
    def resolve_outcome(self):
        """Gets all hands, gets all handranks,finds and counts lowest (strongest) hands. Assigns the pot to them."""
        hands,positions = self.players.return_active_hands()
        if len(positions) > 1:
            # encode hands
            en_hands = []
            for hand in hands:
                en_hand = [encode(c) for c in hand]
                en_hands.append(en_hand)
            en_board = [encode(self.board[i*2:(i+1)*2]) for i in range(0,len(self.board)//2)]
            hand_ranks = [hand_rank(hand,en_board) for hand in en_hands]
            best_hand = np.min(hand_ranks)
            winner_mask = np.where(best_hand == hand_ranks)[0]
            winner_positions = np.array(positions)[winner_mask]
            for winner in winner_positions:
                self.players[winner].stack += self.pot / len(winner_mask)
        else:
            self.players[positions[0]].stack += self.pot

    def return_masks(self,state):
        """
        Grabs last action, return mask of action possibilities. Requires categorical action selection.
        Knows which actions represent bets and their size. When it is a bet, checks stack sizes to make sure betting/calling
        etc. are valid. If both players are allin, then game finishes.
        """
        available_categories = self.return_mask()
        available_betsizes = self.return_betsizes(state)
        if available_betsizes.sum() == 0:
            available_categories[pdt.Globals.REVERSE_ACTION_ORDER[pdt.Actions.RAISE]] = 0
        return available_categories,available_betsizes
    
    def return_mask(self):
        return copy.copy(self.mask_dict[self.global_states.last_aggressive_action])

    def convert_to_category(self,action,betsize):
        """returns int"""
        category = np.zeros(self.action_space + self.betsize_space - 2)
        bet_category = np.zeros(self.betsize_space)
        if action == 0 or action == 1 or action == 2: # fold check call
            category[action] = 1
            bet_category[0] = 1
        elif action == 4:
            if self.global_states.last_aggressive_action == pdt.Globals.REVERSE_ACTION_ORDER[pdt.Actions.RAISE]:
                min_raise = min((self.players[self.current_player].stack+self.players[self.current_player].street_total),max(2,(2*self.players[self.last_aggressor.key].street_total) - self.players[self.current_player].street_total))
            else:
                min_raise = 2 * self.global_states.last_aggressive_betsize
            max_raise = min((2 * self.players[self.last_aggressor.key].street_total) + (self.pot - self.players[self.current_index.key].street_total),(self.players[self.current_player].stack+self.players[self.current_player].street_total))
            previous_bet = self.players[self.last_aggressor.key].street_total - self.players[self.current_index.key].street_total
            possible_sizes = np.linspace(min_raise,max_raise,self.num_betsizes)
            closest_val = np.min(np.abs(possible_sizes - betsize))
            bet_index = np.where(closest_val == np.abs(possible_sizes - betsize))[0]
            bet_category[bet_index] = 1
            category[bet_index+3] = 1
        else:
            max_bet = self.pot
            min_bet = 1
            possible_sizes = self.betsizes * self.pot
            closest_val = np.min(np.abs(possible_sizes - betsize))
            bet_index = np.where(closest_val == np.abs(possible_sizes - betsize))[0]
            category[bet_index+3] = 1
            bet_category[bet_index] = 1
        return int(np.where(category == 1)[0][0]),int(np.where(bet_category == 1)[0][0])

################################################
#                  BETSIZES                    #
################################################

    def return_betsizes(self,state):
        """Returns possible betsizes. If betsize > player stack no betsize is possible"""
        # STREAMLINE THIS
        possible_betsizes = np.zeros((self.num_betsizes))
        if self.global_states.last_aggressive_betsize > 0:
            # Facing a Raise
            if self.global_states.last_aggressive_action == pdt.Globals.REVERSE_ACTION_ORDER[pdt.Actions.RAISE]:
                last_betsize = self.players[self.last_aggressor.key].street_total - self.players[self.current_player].street_total
            else:
                last_betsize = self.global_states.last_aggressive_betsize
            if last_betsize < self.players[self.current_player].stack:
                # player allin is always possible if stack > betsize.
                min_raise = 2 * last_betsize
                max_raise = self.pot + last_betsize
                for i,betsize in enumerate(self.betsizes,0):
                    possible_betsizes[i] = 1
                    if i == 0 and min_raise >= self.players[self.current_player].stack:
                        break
                    elif max_raise * betsize >= self.players[self.current_player].stack:
                        break
        elif self.global_states.last_aggressive_action == 5 or self.global_states.last_aggressive_action == 0:
            for i,betsize in enumerate(self.betsizes,0):
                possible_betsizes[i] = 1
                if betsize * self.pot >= self.players[self.current_player].stack:
                    break
        return possible_betsizes

    ## LIMIT ##
    def return_limit_betsize(self,action,betsize_category):
        """TODO Betsizes should be indexed by street"""
        assert isinstance(action,int)
        assert isinstance(betsize_category,int)
        if action == pdt.Globals.REVERSE_ACTION_ORDER[pdt.Actions.CALL]: # Call
            betsize = min(self.players[self.current_player].stack,self.players[self.last_aggressor.key].street_total - self.players[self.current_player].street_total)
        elif action == pdt.Globals.REVERSE_ACTION_ORDER[pdt.Actions.BET]: # Bet
            betsize = min(1,self.players[self.current_player].stack)
        elif action == pdt.Globals.REVERSE_ACTION_ORDER[pdt.Actions.RAISE]: # Raise can be more than 2 with multiple people
            betsize = min(self.players[self.last_aggressor.key].street_total+1,self.players[self.current_player].stack) - self.players[self.current_player].street_total
        else: # fold or check
            betsize = 0
        return betsize

    ## NO LIMIT ##
    def return_nolimit_betsize(self,action,betsize_category):
        """
        TODO
        Betsize_category would make the most sense if it represented percentages of pot on the first portion.
        And percentages of stack or simply an allin option as the last action.
        """
        assert isinstance(action,int)
        assert isinstance(betsize_category,int)
        if action == pdt.Globals.REVERSE_ACTION_ORDER[pdt.Actions.CALL]:
            betsize = min(self.players[self.last_aggressor.key].street_total - self.players[self.current_index.key].street_total,self.players[self.current_player].stack)
        elif action == pdt.Globals.REVERSE_ACTION_ORDER[pdt.Actions.BET]: # Bet
            betsize_value = self.betsizes[betsize_category] * self.pot
            betsize = min(max(1,betsize_value),self.players[self.current_player].stack)
        elif action == pdt.Globals.REVERSE_ACTION_ORDER[pdt.Actions.RAISE]: # Raise
            max_raise = (2 * self.players[self.last_aggressor.key].street_total) + (self.pot - self.players[self.current_index.key].street_total)
            betsize_value = self.betsizes[betsize_category] * max_raise
            previous_bet = self.players[self.last_aggressor.key].street_total - self.players[self.current_index.key].street_total
            betsize = min(max(previous_bet * 2,betsize_value),self.players[self.current_player].stack)
        else:
            betsize = 0
        return betsize
            
    ## POT LIMIT
    def return_potlimit_betsize(self,action:int,betsize_category:int):
        """TODO Betsize_category in POTLIMIT is a float [0,1] representing fraction of pot"""
        assert isinstance(action,int)
        assert isinstance(betsize_category,int)
        if action == pdt.Globals.REVERSE_ACTION_ORDER[pdt.Actions.CALL]:
            betsize = min(self.players[self.last_aggressor.key].street_total - self.players[self.current_index.key].street_total,self.players[self.current_player].stack)
        elif action == pdt.Globals.REVERSE_ACTION_ORDER[pdt.Actions.BET]: # Bet
            betsize_value = self.betsizes[betsize_category] * self.pot
            betsize = min(max(1,betsize_value),self.players[self.current_player].stack)
        elif action == pdt.Globals.REVERSE_ACTION_ORDER[pdt.Actions.RAISE]: # Raise
            max_raise = (2 * self.players[self.last_aggressor.key].street_total) + (self.pot - self.players[self.current_index.key].street_total)
            betsize_value = (self.betsizes[betsize_category] * max_raise) - self.players[self.current_index.key].street_total
            previous_bet = self.players[self.last_aggressor.key].street_total - self.players[self.current_index.key].street_total
            betsize = min(max(previous_bet * 2,betsize_value),self.players[self.current_player].stack)
        else:
            betsize = 0
        return betsize

    def player_rewards(self):
        rewards = {}
        for position in ['SB','BB']:
            rewards[position] = self.players[position].stack - self.starting_stack
        return rewards
    
    @property
    def current_player(self):
        return self.current_index.key

    def current_stack(self):
        return self.players[self.current_player].stack
    
    @property
    def action_space(self):
        return 5
    # return specs.DiscreteArray(
    #     dtype=int, num_values=len(_ACTIONS), name="action")
    
    @property
    def state_space(self):
        return 31 + self.n_players * 4
        # return BoundedArray(shape=self._board.shape, dtype=self._board.dtype,
        #                       name="state", minimum=0, maximum=1)
    
    @property
    def betsize_space(self):
        return len(self.betsizes)
        # return BoundedArray(shape=self._board.shape, dtype=self._board.dtype,
        #                       name="betsize", minimum=0, maximum=1)
    
    @property
    def observation_space(self):
        return 21 + self.n_players * 4 + self.n_players * 10
        # return BoundedArray(shape=self._board.shape, dtype=self._board.dtype,
        #                       name="observation", minimum=0, maximum=1)

    @property
    def global_space(self):
        return 21 + self.n_players * 4