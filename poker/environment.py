from data_classes import Rules,Evaluator,Pot,GameTurn,Deck,Players,History
import torch
import pickle
import copy

class Poker(object):
    def __init__(self,params):
        self.params = params
        self.game = params['game']
        print(f'Initializating poker game {self.game}')
        self.ranks,self.suits = Poker.card_types(self.game)
        params['state_params']['ranks'] = self.ranks
        params['state_params']['suits'] = self.suits
        self.state_params = params['state_params']
        self.evaluator = Evaluator(self.game)
        self.rules = Rules(params)
        # State attributes
        self.stacksize = self.state_params['stacksize']
        self.n_players = self.state_params['n_players']
        self.pot = Pot(self.state_params['pot'])
        self.game_turn = GameTurn(self.state_params['game_turn'])
        self.stacksizes = torch.Tensor(self.n_players).fill_(self.stacksize)
        self.deck = Deck(self.state_params['ranks'],self.state_params['suits'])
        self.deck.shuffle()
        hands = self.deck.deal(self.state_params['cards_per_player'] * self.n_players)
        self.players = Players(self.n_players,self.stacksizes,hands,self.state_params['to_act'])
        self.history = History()
        
    @staticmethod
    def card_types(game):
        if game == 'kuhn':
            ranks = list(range(1,4))
            suits = None
        return ranks,suits
    
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
        hands = self.deck.deal(self.state_params['cards_per_player'] * self.n_players)
        self.players.reset(hands)
        state,obs = self.return_state()
        self.players.store_states(state,obs)
        return state,obs,self.game_over
    
    def increment_turn(self):
        self.players.increment()
        self.game_turn.increment()

    def step(self,action,action_logprobs):
        self.players.store_actions(action,action_logprobs)
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
    
    def determine_winner(self):
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
            
    def return_state(self,action=None):
        """
        Current player's hand. Previous action
        """
        current_hand = torch.tensor([self.players.current_hand.rank])
        if action == 0:
            state = torch.Tensor([0])
        elif action == 1:
            state = torch.Tensor([1])
        elif action == 2 or action == 3:
            state = torch.Tensor([-1])
        elif action == None:
            state = torch.Tensor([self.rules.unopened_action])
        else:
            raise ValueError(f'Action type {action} not supported')
        state = torch.cat((current_hand.float(),state)).unsqueeze(0)
        obs = state
        return state,obs
    
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
                assert(isinstance(raw_ml[position]['rewards'][0],torch.Tensor))
                raw_ml[position]['game_states'] = torch.stack(raw_ml[position]['game_states']).view(-1,self.state_space)
                raw_ml[position]['observations'] = torch.stack(raw_ml[position]['observations']).view(-1,self.observation_space)
                raw_ml[position]['actions'] = torch.stack(raw_ml[position]['actions']).view(-1,1)
                raw_ml[position]['action_probs'] = torch.stack(raw_ml[position]['action_probs']).view(-1,1)
                raw_ml[position]['rewards'] = torch.stack(raw_ml[position]['rewards']).view(-1,1)
                # print('rewards',raw_ml[position]['rewards'].size())
                # print('actions',raw_ml[position]['actions'].size())
        # pad if necessary

        return raw_ml
    
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