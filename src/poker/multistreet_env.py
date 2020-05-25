import torch
import pickle
import copy

from poker.data_classes import Rules,Evaluator,Pot,GameTurn,Deck,Players,History
import poker.datatypes as pdt

"""
HU ONLY
Full length games. 
Representation is fixed at Action,Betsize.
Game types: Holdem, Omaha. 
Bet types: Limit, NL, PL.
Params: Game type, Num streets.
Variables: Num cards per player. Num streets.

Step input is a dict containing the actor outputs:
Action,Action_prob,Action_probs
Betsize,(Betsize_prob,Betsize_probs) Optional depending on whether flat or tiered

Step output is
State,Obs,done,Action_mask,Betsize_mask

State,Obs output is stacked gamestate representations

Actor outputs: 
Action_category is always one of the 5 general actions (fold,call,raise,bet,check)
Action is in the shape of nC. Combined actions and betsizes.
Action_probs: Shape (nC)
Action_prob: log_prob of choose the action it picked.
"""

class MSPoker(object):
    def __init__(self,params):
        self.params = params
        self.game = params['game']
        print(f'Initializating poker game {self.game}')
        self.state_params = params['state_params']
        self.evaluator = Evaluator(self.game)
        self.rules = Rules(self.params['rule_params'])
        # State attributes
        self.starting_street = self.params['starting_street'] # Number range(0,4)
        self.street = self.starting_street
        self.n_players = self.state_params['n_players']
        self.stacksize = self.state_params['stacksize']
        self.stacksizes = torch.Tensor(self.n_players).fill_(self.stacksize)
        self.pot = Pot(self.state_params['pot'])
        self.game_turn = GameTurn(self.state_params['game_turn'])
        self.deck = Deck(self.state_params['ranks'],self.state_params['suits'])
        self.deck.shuffle()
        cards = self.deck.deal(self.state_params['cards_per_player'] * self.n_players)
        hands = [cards[player * self.state_params['cards_per_player']:(player+1) * self.state_params['cards_per_player']] for player in range(self.n_players)]
        self.players = Players(self.n_players,self.stacksizes,hands)
        self.board = self.deck.initialize_board(self.starting_street)
        self.history = History()
        self.initialize_functions()
        if self.stacksize < 1:
            raise ValueError('Stacksize must be >= 1')
        self.blinds = [0.5,1]

    def initialize_functions(self):
        betsize_funcs = {
            pdt.LimitTypes.LIMIT : self.return_limit_betsize,
            pdt.LimitTypes.NO_LIMIT : self.return_nolimit_betsize,
            pdt.LimitTypes.POT_LIMIT : self.return_potlimit_betsize,
        }
        self.return_betsize = betsize_funcs[self.rules.bettype]
        # Records action frequencies per street
        self.action_records = {
            0:{i:0 for i in range(5)},
            1:{i:0 for i in range(5)},
            2:{i:0 for i in range(5)},
            3:{i:0 for i in range(5)}
        }
        self.nO = self.return_state()[1].size()[-1]
        self.nS = self.return_state()[0].size()[-1]
    
    def save_state(self,path=None):
        path = self.params['save_path'] if path is None else path
        with open(path, 'wb') as handle:
            pickle.dump(self.params, handle, protocol = pickle.HIGHEST_PROTOCOL)  
    
    def load_env(self,path=None):
        path = self.params['save_path'] if path is None else path
        with open(path, 'rb') as handle:
            b = pickle.load(handle)
        return b
        
    def load_scenario(self,scenario:dict):
        '''
        Game state consists of [Players,Pot,History,Deck]
        '''
        self.history = copy.deepcopy(scenario['history'])
        self.pot = copy.deepcopy(scenario['pot'])
        self.deck = copy.deepcopy(scenario['deck'])
        self.players = copy.deepcopy(scenario['players'])
        self.game_turn = copy.deepcopy(scenario['game_turn'])
        
    def save_scenario(self):
        scenario = {
            'history':copy.deepcopy(self.history),
            'pot':copy.deepcopy(self.pot),
            'deck':copy.deepcopy(self.deck),
            'players':copy.deepcopy(self.players),
            'game_turn':copy.deepcopy(self.game_turn),
        }
        return scenario
        
    def reset(self):
        """
        resets env state to initial state (can be preloaded).
        """
        self.action_records = {
            0:{i:0 for i in range(5)},
            1:{i:0 for i in range(5)},
            2:{i:0 for i in range(5)},
            3:{i:0 for i in range(5)}
        }
        self.history.reset()
        self.pot.reset()
        self.game_turn.reset()
        self.street = self.starting_street
        self.deck.reset()
        self.deck.shuffle()
        cards = self.deck.deal(self.state_params['cards_per_player'] * self.n_players)
        hands = [cards[player * self.state_params['cards_per_player']:(player+1) * self.state_params['cards_per_player']] for player in range(self.n_players)]
        self.players.reset(hands)
        self.players.reset_street_totals()
        if self.street == 0:
            self.initialize_blinds()
        self.board = self.deck.initialize_board(self.starting_street)
        state,obs = self.initialize_state()
        player_state,player_obs = self.return_player_states(state,obs)
        self.players.store_history(player_state,player_obs)
        self.players.store_states(state,obs)
        action_mask,betsize_mask = self.action_mask(state)
        self.players.store_masks(action_mask,betsize_mask)
        return state,obs,self.game_over,action_mask,betsize_mask

    def initialize_state(self):
        """
        (will eventually need active, allin)
        STATE: P1 Hand,Board,street,P1 position,P2 position,Previous_action,Previous_betsize,P1 Stack,P2 Stack,Amnt to call,Pot odds
        OBS: P1 Hand,P2 Hand,Board,Street,P1 position,P2 position,Previous_action,Previous_betsize,P1 Stack,P2 Stack,Amnt to call,Pot odds
        Ordinal: P1 position, P2 position, Previous action, Street
        Conv: P1/P2 Hand, Board
        Continuous: P1/P2 stack,Previous_betsize,Amnt to call, Pot odds. (All normalized?)

        Separate representations for each player (SB,BB)
        3 concatenated states each.
        Should be all the positions/stacksizes of currently active players (In order of how they will act?)
        Hero position, stacksize.
        pot
        Previous action,betsize,position of previous player.
        """
        SB_hand = torch.tensor([[card.rank,card.suit] for card in self.players.players[pdt.Positions.SB].hand]).view(-1).squeeze(0).float()
        BB_hand = torch.tensor([[card.rank,card.suit] for card in self.players.players[pdt.Positions.BB].hand]).view(-1).squeeze(0).float()
        
        board_cards = torch.tensor([[card.rank,card.suit] for card in self.board]).view(-1).squeeze(0).float()
        board = torch.Tensor(10).fill_(0.)
        board[:board_cards.size(0)] = board_cards
        street = torch.tensor([self.street]).float()
        SB_position = torch.tensor([0.])
        BB_position = torch.tensor([1.])
        SB_stack = torch.tensor([self.players.players[pdt.Positions.SB].stack]).float()
        BB_stack = torch.tensor([self.players.players[pdt.Positions.BB].stack]).float()
        SB_to_call = torch.tensor([0.])
        SB_pot_odds = torch.tensor([0.])
        if self.street == 0:
            SB_action = torch.tensor([3.])
            SB_betsize = torch.tensor([0.5])
            # Post SB
            SB_state_unopened = torch.cat((SB_hand,board,street,SB_position,BB_position,self.rules.unopened_action,torch.tensor([0.]),SB_stack,BB_stack,SB_to_call,SB_pot_odds)).unsqueeze(0).unsqueeze(0)
            SB_obs_unopened = torch.cat((SB_hand,BB_hand,board,street,SB_position,BB_position,self.rules.unopened_action,torch.tensor([0.]),SB_stack,BB_stack,SB_to_call,SB_pot_odds)).unsqueeze(0).unsqueeze(0)
            BB_state_unopened = torch.cat((BB_hand,board,street,SB_position,BB_position,self.rules.unopened_action,torch.tensor([0.]),SB_stack,BB_stack,SB_to_call,SB_pot_odds)).unsqueeze(0).unsqueeze(0)
            BB_obs_unopened = torch.cat((BB_hand,SB_hand,board,street,SB_position,BB_position,self.rules.unopened_action,torch.tensor([0.]),SB_stack,BB_stack,SB_to_call,SB_pot_odds)).unsqueeze(0).unsqueeze(0)

            SB_stack -= SB_betsize
            # Post BB
            BB_action = torch.tensor([4.])
            BB_betsize = torch.tensor([1.])
            BB_to_call = torch.tensor([0.5])
            BB_pot_odds = torch.tensor([1.])
            BB_state_SB = torch.cat((BB_hand,board,street,BB_position,SB_position,SB_action,SB_betsize,BB_stack,SB_stack,BB_to_call,BB_pot_odds)).unsqueeze(0).unsqueeze(0)
            BB_obs_SB = torch.cat((BB_hand,SB_hand,board,street,BB_position,SB_position,SB_action,SB_betsize,BB_stack,SB_stack,BB_to_call,BB_pot_odds)).unsqueeze(0).unsqueeze(0)
            SB_state_SB = torch.cat((SB_hand,board,street,BB_position,SB_position,SB_action,SB_betsize,BB_stack,SB_stack,BB_to_call,BB_pot_odds)).unsqueeze(0).unsqueeze(0)
            SB_obs_SB = torch.cat((SB_hand,BB_hand,board,street,BB_position,SB_position,SB_action,SB_betsize,BB_stack,SB_stack,BB_to_call,BB_pot_odds)).unsqueeze(0).unsqueeze(0)
            
            BB_stack -= BB_betsize
            # Post SB current state
            current_pot_odds = torch.tensor([0.25])
            SB_current_state = torch.cat((SB_hand,board,street,SB_position,BB_position,BB_action,BB_betsize,SB_stack,BB_stack,BB_to_call,current_pot_odds)).unsqueeze(0).unsqueeze(0)
            SB_current_obs = torch.cat((SB_hand,BB_hand,board,street,SB_position,BB_position,BB_action,BB_betsize,SB_stack,BB_stack,BB_to_call,current_pot_odds)).unsqueeze(0).unsqueeze(0)
            BB_current_state = torch.cat((BB_hand,board,street,SB_position,BB_position,BB_action,BB_betsize,SB_stack,BB_stack,BB_to_call,current_pot_odds)).unsqueeze(0).unsqueeze(0)
            BB_current_obs = torch.cat((BB_hand,SB_hand,board,street,SB_position,BB_position,BB_action,BB_betsize,SB_stack,BB_stack,BB_to_call,current_pot_odds)).unsqueeze(0).unsqueeze(0)
            
            SB_state = torch.cat((SB_state_unopened,SB_state_SB,SB_current_state),dim=1)
            SB_obs = torch.cat((SB_obs_unopened,SB_obs_SB,SB_current_obs),dim=1)

            BB_state = torch.cat((BB_state_unopened,BB_state_SB,BB_current_state),dim=1)
            BB_obs = torch.cat((BB_obs_unopened,BB_obs_SB,BB_current_obs),dim=1)
            self.players.store_states(BB_state,BB_obs,player='BB')
            self.history.add(self.players.players['SB'],SB_action,SB_betsize)
            self.history.add(self.players.players['BB'],BB_action,BB_betsize)
        else:
            SB_state = torch.cat((SB_hand,board,street,SB_position,BB_position,self.rules.unopened_action,torch.tensor([0.]),SB_stack,BB_stack,SB_to_call,SB_pot_odds)).unsqueeze(0).unsqueeze(0)
            SB_obs = torch.cat((SB_hand,BB_hand,board,street,SB_position,BB_position,self.rules.unopened_action,torch.tensor([0.]),SB_stack,BB_stack,SB_to_call,SB_pot_odds)).unsqueeze(0).unsqueeze(0)
            
        return SB_state,SB_obs

    def initialize_blinds(self):
        for position,blind in pdt.Globals.BLIND_DICT.items():
            self.players.update_stack(-blind,player=position)
            self.pot.add(blind)

    def update_board(self,street):
        new_cards = self.deck.deal_board(street)
        for card in new_cards:
            self.board.append(card) 

    def record_action(self,action):
        self.action_records[self.street][action] += 1
    
    def increment_turn(self):
        self.players.increment()
        self.game_turn.increment()

    def increment_street(self):
        self.street += 1
        assert self.street > pdt.Globals.REVERSE_STREET_DICT[pdt.Street.RIVER]
        # clear previous street totals
        self.players.reset_street_totals()
        # update board
        self.update_board(self.street)
        # update positions
        self.players.update_position_order(self.street)
        # Fast forward to river if allin
        if self.players.to_showdown == True:
            for _ in range(pdt.Globals.REVERSE_STREET_DICT[pdt.Street.RIVER] - self.street):
                self.street += 1
                self.update_board(self.street)
                
    def return_full_state(self,state,obs):
        """
        Takes in the last state and concats it with all the states for that player so far. 
        Returns in the form: (b,m,c)
        """
        if len(self.players.game_states[self.players.current_player]) > 0:
            hist_game_states = torch.stack(self.players.game_states[self.players.current_player],dim=0).squeeze(1).permute(1,0,2)
            hist_obs = torch.stack(self.players.observations[self.players.current_player],dim=0).squeeze(1).permute(1,0,2)
            return torch.cat([hist_game_states,state],dim=1),torch.cat([hist_obs,obs],dim=1)
        else:
            return state,obs

    def step(self,actor_outputs):
        self.players.store_actor_outputs(actor_outputs)
        self.update_state(actor_outputs['action_category'],actor_outputs['betsize'])
        state,obs = self.return_state(actor_outputs['action_category'])
        if self.round_over and self.street != pdt.Globals.REVERSE_STREET_DICT[pdt.Street.RIVER]:
            # Check for allins -> accelerate to final
            # deal board, get new state,obs with unopened action, reset player street totals
            self.increment_street()
            new_state,new_obs = self.return_state()
            state = torch.cat([state,new_state])
            obs = torch.cat([obs,new_obs])
        action_mask,betsize_mask = self.action_mask(state)
        full_state,full_obs = self.return_full_state(state,obs)
        done = self.game_over
        if done == False:
            self.players.store_states(state,obs) 
            self.players.store_masks(action_mask,betsize_mask)
        else: 
            self.determine_winner()
        return full_state,full_obs,done,action_mask,betsize_mask
    
    def update_state(self,action,betsize_category=None):
        """Updates the current environment state by processing the current action."""
        action_int = action.item()
        self.record_action(action_int)
        bet_amount = self.return_betsize(action_int,betsize_category)
        self.history.add(self.players.current_player,action_int,bet_amount)
        self.pot.add(bet_amount)
        self.players.update_stack(-bet_amount)
        self.increment_turn()

    def ml_inputs(self):
        """Stacks and returns the hand attributes for training and storage."""
        raw_ml = self.players.get_inputs()
        positions = raw_ml.keys()
        # convert to torch
        for position in positions:
            if position in raw_ml:
                if len(raw_ml[position]['game_states']):
                    assert(isinstance(raw_ml[position]['game_states'][0],torch.Tensor))
                    assert(isinstance(raw_ml[position]['observations'][0],torch.Tensor))
                    assert(isinstance(raw_ml[position]['actions'][0],torch.Tensor))
                    assert(isinstance(raw_ml[position]['action_prob'][0],torch.Tensor))
                    assert(isinstance(raw_ml[position]['action_probs'][0],torch.Tensor))
                    assert(isinstance(raw_ml[position]['rewards'][0],torch.Tensor))
                    assert(isinstance(raw_ml[position]['historical_game_states'][0],torch.Tensor))
                    
                    raw_ml[position]['historical_game_states'] = torch.stack(raw_ml[position]['historical_game_states']).view(-1,self.maxlen,self.state_space)
                    raw_ml[position]['hand_strength'] = raw_ml[position]['hand_strength']
                    raw_ml[position]['game_states'] = torch.stack(raw_ml[position]['game_states']).view(1,-1,self.state_space)
                    raw_ml[position]['observations'] = torch.stack(raw_ml[position]['observations']).view(1,-1,self.observation_space)
                    raw_ml[position]['actions'] = torch.stack(raw_ml[position]['actions']).view(1,-1,1)
                    raw_ml[position]['action_prob'] = torch.stack(raw_ml[position]['action_prob']).view(1,-1,1)
                    raw_ml[position]['rewards'] = torch.stack(raw_ml[position]['rewards']).view(1,-1,1)
                    raw_ml[position]['dones'] = torch.stack(raw_ml[position]['dones']).view(1,-1,1)
                    raw_ml[position]['action_masks'] = torch.stack(raw_ml[position]['action_masks']).view(1,-1,self.action_space)
                    raw_ml[position]['betsize_masks'] = torch.stack(raw_ml[position]['betsize_masks']).view(1,-1,self.betsize_space)

                    if self.rules.network_output == 'flat':
                        raw_ml[position]['betsizes'] = torch.stack(raw_ml[position]['betsizes']).view(1,-1,1)
                        raw_ml[position]['action_probs'] = torch.stack(raw_ml[position]['action_probs']).view(1,-1,self.action_space - 2 + self.betsize_space)
                    else:
                        raw_ml[position]['betsizes'] = torch.stack(raw_ml[position]['betsizes']).view(1,-1,1)
                        raw_ml[position]['betsize_prob'] = torch.stack(raw_ml[position]['betsize_prob']).view(1,-1,1)
                        raw_ml[position]['betsize_probs'] = torch.stack(raw_ml[position]['betsize_probs']).view(1,-1,self.betsize_space)
                        raw_ml[position]['action_probs'] = torch.stack(raw_ml[position]['action_probs']).view(1,-1,self.action_space)
        return raw_ml
        
    @property
    def db_mapping(self):
        """Required for inserting the data into mongodb properly"""
        return self.rules.db_mapping

    @property
    def round_over(self):
        return self.rules.over(self)

    @property
    def game_over(self):
        return self.rules.over(self) and self.street == pdt.Globals.REVERSE_STREET_DICT[pdt.Street.RIVER]
    
    @property
    def state_space(self):
        return self.nS
    
    @property
    def observation_space(self):
        return self.nO
    
    @property
    def action_space(self):
        return self.rules.action_space

    @property
    def betsize_space(self):
        return self.rules.num_betsizes

    @property
    def current_player(self):
        return self.players.current_player

    @property
    def previous_player(self):
        return self.players.previous_player

    def determine_outcome(self):
        """Determines winner, allots pot to winner, records handstrengths for each player"""
        self.players.store_handstrengths(self.board)
        if self.history.last_action == pdt.Globals.REVERSE_ACTION_ORDER[pdt.Actions.FOLD]:
            self.players.update_stack(self.pot.value)
        else:
            hands = self.players.get_hands()
            hand1,hand2 = hands
            winner_idx = self.evaluator([hand1,hand2,self.board])
            if winner_idx == 1:
                self.players.update_stack(self.pot.value,'SB')
            elif winner_idx == -1:
                self.players.update_stack(self.pot.value,'BB')
            else:
                # Tie
                self.players.update_stack(self.pot.value / 2,'SB')
                self.players.update_stack(self.pot.value / 2,'BB')
        self.players.gen_rewards()

    def return_state(self,action=None):
        """
        (will eventually need active, allin)
        STATE: P1 Hand,Board,street,P1 position,P2 position,Previous_action,Previous_betsize,P1 Stack,P2 Stack,Amnt to call,Pot odds
        OBS: P1 Hand,P2 Hand,Board,Street,P1 position,P2 position,Previous_action,Previous_betsize,P1 Stack,P2 Stack,Amnt to call,Pot odds
        Ordinal: P1 position, P2 position, Previous action, Street
        Conv: P1/P2 Hand, Board
        Continuous: P1/P2 stack,Previous_betsize,Amnt to call, Pot odds. (All normalized?)
        """
        if not isinstance(action,torch.Tensor):
            action = torch.Tensor([self.rules.unopened_action])
        current_hand = torch.tensor([[card.rank,card.suit] for card in self.players.current_hand]).view(-1).squeeze(0).float()
        vil_hand = torch.tensor([[card.rank,card.suit] for card in self.players.previous_hand]).view(-1).squeeze(0).float()
        board_cards = torch.tensor([[card.rank,card.suit] for card in self.board]).view(-1).squeeze(0).float()
        board = torch.Tensor(10).fill_(0.)
        board[:board_cards.size(0)] = board_cards
        street = torch.tensor([self.street]).float()

        current_position = torch.tensor([pdt.Globals.POSITION_MAPPING[self.players.current_player]]).float()
        vil_position = torch.tensor([pdt.Globals.POSITION_MAPPING[self.players.previous_player]]).float()
        ## If the last betsize is 0 then both are zero
        if self.history.last_betsize > 0:
            to_call = self.players.previous_street_total - self.players.current_street_total
            pot_odds = to_call / (self.pot.value + to_call) # Because state is updated prior to retriving gamestate. The bet has already been added to the pot
        else:
            to_call = torch.tensor([0])
            pot_odds = torch.tensor([0])
        state = torch.cat((current_hand,board,street,current_position,vil_position,torch.tensor([self.history.last_action]).float(),self.history.last_betsize.float(),self.players.current_stack.unsqueeze(-1).float(),self.players.previous_stack.unsqueeze(-1).float(),to_call.float(),pot_odds.float())).unsqueeze(0).unsqueeze(0)
        obs = torch.cat((current_hand,vil_hand,board,street,current_position,vil_position,torch.tensor([self.history.last_action]).float(),self.history.last_betsize.float(),self.players.current_stack.unsqueeze(-1).float(),self.players.previous_stack.unsqueeze(-1).float(),to_call.float(),pot_odds.float())).unsqueeze(0).unsqueeze(0)
        return state,obs

    def action_mask(self,state):
        """
        Called from outside when training.
        Grabs last action, return mask of action possibilities. Requires categorical action selection.
        Knows which actions represent bets and their size. When it is a bet, checks stack sizes to make sure betting/calling
        etc. are valid. If both players are allin, then game finishes.
        """
        available_categories = self.rules.return_mask(state)
        available_betsizes = self.return_betsizes(state)
        if available_betsizes.sum() == 0:
            available_categories[pdt.Globals.REVERSE_ACTION_ORDER[pdt.Actions.RAISE]] = 0
        return available_categories,available_betsizes

################################################
#                  BETSIZES                    #
################################################

    def return_betsizes(self,state):
        """Returns possible betsizes. If betsize > player stack no betsize is possible"""
        # STREAMLINE THIS
        possible_betsizes = torch.zeros((self.rules.num_betsizes))
        if self.history.last_betsize > 0:
            if self.history.last_action == pdt.Globals.REVERSE_ACTION_ORDER[pdt.Actions.RAISE]:
                previous_betsize = self.history.last_betsize - self.history.penultimate_betsize
            else:
                previous_betsize = self.history.last_betsize
            if previous_betsize < self.players.current_stack:
                # player allin is always possible if stack > betsize.
                for i,betsize in enumerate(self.rules.betsizes,0):
                    possible_betsizes[i] = 1
                    if betsize * self.pot.value >= self.players.current_stack:
                        break
        elif state[-1,-1,self.rules.db_mapping['state']['previous_action']] == 5 or state[-1,-1,self.rules.db_mapping['state']['previous_action']] == 0:
            for i,betsize in enumerate(self.rules.betsizes,0):
                possible_betsizes[i] = 1
                if betsize * self.pot.value >= self.players.current_stack:
                    break
        return possible_betsizes

    ## LIMIT ##
    def return_limit_betsize(self,action,betsize_category):
        """TODO Betsizes should be indexed by street"""
        if action == pdt.Globals.REVERSE_ACTION_ORDER[pdt.Actions.CALL]: # Call
            betsize = min(1,self.players.current_stack)
        elif action == pdt.Globals.REVERSE_ACTION_ORDER[pdt.Actions.BET]: # Bet
            betsize = min(1,self.players.current_stack)
        elif action == pdt.Globals.REVERSE_ACTION_ORDER[pdt.Actions.RAISE]: # Raise
            betsize = min(2,self.players.current_stack)
        else: # fold or check
            betsize = 0
        return torch.tensor([betsize])

    ## NO LIMIT ##
    def return_nolimit_betsize(self,action,betsize_category):
        """Betsize_category in NL is a float [0,1] representing percentage of pot"""
        if action > 1:
            if action == pdt.Globals.REVERSE_ACTION_ORDER[pdt.Actions.CALL]:
                betsize = min(self.history.last_betsize - self.history.penultimate_betsize,self.players.current_stack)
            elif action == pdt.Globals.REVERSE_ACTION_ORDER[pdt.Actions.BET]: # Bet
                betsize_value = self.rules.betsizes[betsize_category.long()] * self.pot.value
                betsize = min(max(self.rules.minbet,betsize_value),self.players.current_stack)
                # print('betsize_value',betsize_value)
            elif action == pdt.Globals.REVERSE_ACTION_ORDER[pdt.Actions.RAISE]: # Raise
                max_raise = (2 * self.players.previous_street_total) + (self.pot.value - self.players.current_street_total)
                betsize_value = self.rules.betsizes[betsize_category.long()] * max_raise
                previous_bet = self.players.previous_street_total - self.players.current_street_total
                betsize = min(max(previous_bet * 2,betsize_value),self.players.current_stack)
                # print('betsize_value',betsize_value)
        else:
            betsize = 0
        # print('self.players.current_stack',self.players.current_stack)
        # print('action',action)
        # print('betsize',betsize)
        return torch.tensor([betsize])
            
    ## POT LIMIT
    def return_potlimit_betsize(self,action,betsize_category):
        """TODO Betsize_category in POTLIMIT is a float [0,1] representing fraction of pot"""
        if action > 2:
            min_betsize = self.rules.minbet if action == 3 else self.history.last_betsize * 2
            betsize = min(max(min_betsize,self.rules.betsizes[betsize_category.long()] * self.pot.value),self.players.current_stack)
        else:
            betsize = 0
        print('betsize',betsize)
        return torch.tensor([betsize])
        