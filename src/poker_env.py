import torch
import pickle
import copy

from poker.data_classes import Rules,Evaluator,Pot,GameTurn,Deck,Players,History
import poker.datatypes as pdt

class Poker(object):
    def __init__(self,params):
        self.params = params
        self.game = params['game']
        print(f'Initializating poker game {self.game}')
        self.state_params = params['state_params']
        self.ranks = params['state_params']['ranks']
        self.suits = params['state_params']['suits']
        self.evaluator = Evaluator(self.game)
        self.rules = Rules(self.params['rule_params'])
        # State attributes
        self.stacksize = self.state_params['stacksize']
        self.n_players = self.state_params['n_players']
        self.pot = Pot(self.state_params['pot'])
        self.game_turn = GameTurn(self.state_params['game_turn'])
        self.stacksizes = torch.Tensor(self.n_players).fill_(self.stacksize)
        self.deck = Deck(self.state_params['ranks'],self.state_params['suits'])
        self.deck.shuffle()
        cards = self.deck.deal(self.state_params['cards_per_player'] * self.n_players)
        hands = [cards[player * self.state_params['cards_per_player']:(player+1) * self.state_params['cards_per_player']] for player in range(self.n_players)]
        self.players = Players(self.n_players,self.stacksizes,hands,self.state_params['to_act'])
        if self.suits != None:
            self.board = self.deck.deal(5)
        self.history = History()
        self.initialize_functions()

    def initialize_functions(self):
        func_dict = {
            pdt.GameTypes.KUHN : {'determine_winner':self.determine_kuhn,'return_state':self.return_kuhn_state},
            pdt.GameTypes.COMPLEXKUHN : {'determine_winner':self.determine_kuhn,'return_state':self.return_kuhn_state},
            pdt.GameTypes.HOLDEM : {'determine_winner':self.determine_holdem,'return_state':self.return_holdem_state},
            pdt.GameTypes.OMAHAHI : 'not implemented'
        }
        self.determine_winner = func_dict[self.game]['determine_winner']
        self.return_state = func_dict[self.game]['return_state']
    
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
        self.history.reset()
        self.pot.reset()
        self.game_turn.reset()
        self.deck.reset()
        self.deck.shuffle()
        cards = self.deck.deal(self.state_params['cards_per_player'] * self.n_players)
        hands = [cards[player * self.state_params['cards_per_player']:(player+1) * self.state_params['cards_per_player']] for player in range(self.n_players)]
        self.players.reset(hands)
        state,obs = self.return_state()
        self.players.store_states(state,obs)
        if self.suits != None:
            self.board = self.deck.deal(5)
        return state,obs,self.game_over
    
    def increment_turn(self):
        self.players.increment()
        self.game_turn.increment()

    def step(self,action,action_logprobs,complete_probs):
        self.players.store_actions(action,action_logprobs,complete_probs)
        self.update_state(action)
        state,obs = self.return_state(action)
        done = self.game_over
        if done == False:
            self.players.store_states(state,obs)
        else:
            self.determine_winner()
        return state,obs,done
    
    def update_state(self,action):
        action_int = action.item()
        bet_amount = self.rules.betsize_dict[action_int]
        self.history.add(self.players.current_player,action_int)
        self.pot.add(bet_amount)
        self.players.update_stack(-bet_amount)
        self.increment_turn()
    
    def ml_inputs(self):
        raw_ml = self.players.get_inputs()
        positions = raw_ml.keys()
        # convert to torch
        for position in positions:
            if len(raw_ml[position]['game_states']):
                assert(isinstance(raw_ml[position]['game_states'][0],torch.Tensor))
                assert(isinstance(raw_ml[position]['observations'][0],torch.Tensor))
                assert(isinstance(raw_ml[position]['actions'][0],torch.Tensor))
                assert(isinstance(raw_ml[position]['action_probs'][0],torch.Tensor))
                assert(isinstance(raw_ml[position]['complete_probs'][0],torch.Tensor))
                assert(isinstance(raw_ml[position]['rewards'][0],torch.Tensor))
                # assert(isinstance(raw_ml[position]['values'][0],torch.Tensor))
                raw_ml[position]['game_states'] = torch.stack(raw_ml[position]['game_states']).view(-1,self.state_space)
                raw_ml[position]['observations'] = torch.stack(raw_ml[position]['observations']).view(-1,self.observation_space)
                raw_ml[position]['actions'] = torch.stack(raw_ml[position]['actions']).view(-1,1)
                raw_ml[position]['action_probs'] = torch.stack(raw_ml[position]['action_probs']).view(-1,1)
                raw_ml[position]['complete_probs'] = torch.stack(raw_ml[position]['complete_probs']).view(-1,self.action_space)
                raw_ml[position]['rewards'] = torch.stack(raw_ml[position]['rewards']).view(-1,1)
                # raw_ml[position]['values'] = torch.stack(raw_ml[position]['values']).view(-1,1)
        return raw_ml
    
    def action_mask(self,state):
        return self.rules.return_mask(state)

    @property
    def db_mapping(self):
        """
        Required for inserting the data into mongodb properly
        """
        return self.rules.db_mapping

    @property
    def game_over(self):
        return self.rules.over(self)
    
    @property
    def state_space(self):
        return self.return_state()[0].size()[-1]
    
    @property
    def observation_space(self):
        return self.return_state()[1].size()[-1]
    
    @property
    def action_space(self):
        return self.rules.action_space

    @property
    def current_player(self):
        return self.players.current_player

    def determine_holdem(self):
        if self.history.last_action == 3:
            self.players.update_stack(self.pot.value)
        else:
            hands = self.players.get_hands()
            hand1,hand2 = hands
            winner_idx = self.evaluator([hand1,hand2,self.board])
            winner_position = self.players.initial_positions[winner_idx]
            self.players.update_stack(self.pot.value,winner_position)
        self.players.gen_rewards()

    def determine_kuhn(self):
        """
        Two cases, showdown and none showdown.
        """
        if self.history.last_action == 3:
            self.players.update_stack(self.pot.value)
        else:
            hands = self.players.get_hands()
            winner_idx = self.evaluator(hands)
            winner_position = self.players.initial_positions[winner_idx]
            self.players.update_stack(self.pot.value,winner_position)
        self.players.gen_rewards()

    def return_holdem_state(self,action=None):
        """
        Current player's hand. Previous action
        """
        current_hand = torch.tensor([[card.rank,card.suit] for card in self.players.current_hand]).view(-1).squeeze(0)
        vil_hand = torch.tensor([[card.rank,card.suit] for card in self.players.previous_hand]).view(-1).squeeze(0)
        board = torch.tensor([[card.rank,card.suit] for card in self.board]).view(-1).squeeze(0)
        if not isinstance(action,torch.Tensor):
            action = torch.Tensor([self.rules.unopened_action])
        state = torch.cat((current_hand.float(),board.float(),action.float())).unsqueeze(0)
        # Obs
        obs = torch.cat((current_hand.float(),vil_hand.float(),board.float(),action.float())).unsqueeze(0)
        return state,obs

    def return_kuhn_state(self,action=None):
        """
        Current player's hand. Previous action
        """
        current_hand = torch.tensor([card.rank for card in self.players.current_hand])
        vil_hand = torch.tensor([card.rank for card in self.players.previous_hand])
        if not isinstance(action,torch.Tensor):
            action = torch.Tensor([self.rules.unopened_action])
        state = torch.cat((current_hand.float(),action.float())).unsqueeze(0)
        # Obs
        obs = torch.cat((current_hand.float(),vil_hand.float(),action.float())).unsqueeze(0)
        return state,obs

    def available_actions(self):
        """
        Grabs last action, return mask of action possibilities. Requires categorical action selection.
        Knows which actions represent bets and their size. When it is a bet, checks stack sizes to make sure betting/calling
        etc. are valid. If both players are allin, then game finishes.
        """
        current_player = self.players.current_player
        current_stack = current_player.stack
        # last_action
        # Categorical actions
        # betsizes
        
        