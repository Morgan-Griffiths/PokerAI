
import numpy as np
import torch
import unittest
import numpy as np
import os
import copy

from models.networks import OmahaActor,OmahaQCritic,OmahaObsQCritic,CombinedNet
from poker_env.env import Poker,Status
from poker_env.config import Config
import poker_env.datatypes as pdt
from utils.cardlib import winner,holdem_winner,encode

ACTION_CHECK = {
    'action':0,
    'action_category':0,
    'action_probs':torch.zeros(5).fill_(0.2),
    'action_prob':np.array([0.2]),
    'betsize':0
}
ACTION_FOLD = {
    'action':1,
    'action_category':1,
    'action_probs':torch.zeros(5).fill_(0.2),
    'action_prob':np.array([0.2]),
    'betsize':0
}
ACTION_BET = {
    'action':4,
    'action_category':3,
    'action_probs':torch.zeros(5).fill_(0.2),
    'action_prob':np.array([0.2]),
    'betsize':1
}
ACTION_CALL = {
    'action':2,
    'action_category':2,
    'action_probs':torch.zeros(5).fill_(0.2),
    'action_prob':np.array([0.2]),
    'betsize':0
}
ACTION_RAISE = {
    'action':4,
    'action_category':4,
    'action_probs':torch.zeros(5).fill_(0.2),
    'action_prob':np.array([0.2]),
    'betsize':1
}
ACTION_MIN_RAISE = {
    'action':4,
    'action_category':4,
    'action_probs':torch.zeros(5).fill_(0.2),
    'action_prob':np.array([0.2]),
    'betsize':0
}

STATE_SHAPE = 39
OBS_SHAPE = 49

def run_poker_env(env,case):
    step = 0
    state,obs,done,mask,betsize_mask = env.reset()
    while not done:
        action,action_category,action_prob,action_probs = case[step]
        actor_output = {
            'action':action,
            'action_category':action_category,
            'action_prob':action_prob,
            'action_probs':action_probs
        }
        state,obs,done,mask,betsize_mask = env.step(actor_output)
        step += 1
    return env

class TestEnv(unittest.TestCase):
    @classmethod
    def setUp(self):
        game_object = pdt.Globals.GameTypeDict[pdt.GameTypes.OMAHAHI]
        config = Config()
        self.env_params = {
            'game':pdt.GameTypes.OMAHAHI,
            'betsizes': game_object.rule_params['betsizes'],
            'bet_type': game_object.rule_params['bettype'],
            'n_players': 2,
            'pot':1,
            'stacksize': 5.,
            'cards_per_player': game_object.state_params['cards_per_player'],
            'starting_street': game_object.starting_street,
            'global_mapping':config.global_mapping,
            'state_mapping':config.state_mapping,
            'obs_mapping':config.obs_mapping,
            'shuffle':False
        }

    def testInitialization(self):
        env = Poker(self.env_params)
        assert env.street == self.env_params['starting_street']
        assert env.game == self.env_params['game']
        assert env.n_players == self.env_params['n_players']
        assert len(env.players) == self.env_params['n_players']
        assert env.starting_stack == self.env_params['stacksize']

    def testReset(self):
        params = copy.deepcopy(self.env_params)
        params['starting_street'] = pdt.Street.RIVER
        env = Poker(params)
        state,obs,done,mask,betsize_mask = env.reset()
        assert state.ndim == 3
        assert obs.ndim == 3
        assert state.shape == (1,1,STATE_SHAPE)
        assert obs.shape == (1,1,OBS_SHAPE)
        assert state[0,0,env.state_mapping['street']] == pdt.Street.RIVER
        assert state[0,-1,env.state_mapping['hero_position']] == pdt.Position.BB
        assert state[0,-1,env.state_mapping['hero_stacksize']] == self.env_params['stacksize']
        assert state[0,-1,env.state_mapping['player1_position']] == pdt.Position.BB
        assert state[0,-1,env.state_mapping['player1_stacksize']] == self.env_params['stacksize']
        assert state[0,-1,env.state_mapping['player1_street_total']] == 0
        assert state[0,-1,env.state_mapping['player2_position']] == pdt.Position.SB
        assert state[0,-1,env.state_mapping['player2_stacksize']] == self.env_params['stacksize']
        assert state[0,-1,env.state_mapping['player2_street_total']] == 0
        assert state[0,-1,env.state_mapping['last_action']] == pdt.Action.UNOPENED
        assert state[0,-1,env.state_mapping['last_aggressive_action']] == pdt.Action.UNOPENED
        assert state[0,-1,env.state_mapping['last_betsize']] == 0
        assert state[0,-1,env.state_mapping['last_position']] == pdt.Position.BTN
        assert state[0,-1,env.state_mapping['amount_to_call']] == 0
        assert state[0,-1,env.state_mapping['pot_odds']] == 0
        assert env.players_remaining == 2
        assert done == False
        assert np.array_equal(mask,np.array([1., 0., 0., 1., 0.]))
        assert np.array_equal(betsize_mask,np.array([1.,1.]))
        assert len(env.players.players['SB'].hand) == self.env_params['cards_per_player']
        assert len(env.players.players['BB'].hand) == self.env_params['cards_per_player']
        assert len(env.deck) == 52 - (self.env_params['cards_per_player'] * self.env_params['n_players'] + pdt.Globals.INITIALIZE_BOARD_CARDS[self.env_params['starting_street']]) 

    def testCheckCheck(self):
        env = Poker(self.env_params)
        state,obs,done,mask,betsize_mask = env.reset()
        state,obs,done,mask,betsize_mask = env.step(ACTION_CHECK)
        assert state.ndim == 3
        assert obs.ndim == 3
        assert state.shape == (1,2,STATE_SHAPE)
        assert obs.shape == (1,2,OBS_SHAPE)
        assert state[:,1][:,env.state_mapping['street']] == self.env_params['starting_street']
        assert state[:,1][:,env.state_mapping['hero_position']] == pdt.Position.SB
        assert state[:,1][:,env.state_mapping['player2_position']] == pdt.Position.BB
        assert state[:,1][:,env.state_mapping['last_position']] == pdt.Position.BB
        assert state[:,1][:,env.state_mapping['last_action']] == pdt.Action.CHECK
        assert state[:,1][:,env.state_mapping['hero_stacksize']] == self.env_params['stacksize']
        assert state[:,1][:,env.state_mapping['player2_stacksize']] == self.env_params['stacksize']
        assert state[:,1][:,env.state_mapping['amount_to_call']] == 0
        assert state[:,1][:,env.state_mapping['pot_odds']] == 0
        assert done == False
        assert np.array_equal(mask,np.array([1., 0., 0., 1., 0.]))
        assert np.array_equal(betsize_mask,np.array([1.,1.]))

        state,obs,done,mask,betsize_mask = env.step(ACTION_CHECK)
        assert done == True
        assert env.players['SB'].stack == 6
        assert env.players['BB'].stack == 5

    def testCheckBetFold(self):
        env = Poker(self.env_params)
        state,obs,done,mask,betsize_mask = env.reset()
        state,obs,done,mask,betsize_mask = env.step(ACTION_CHECK)
        state,obs,done,mask,betsize_mask = env.step(ACTION_BET)
        assert state.ndim == 3
        assert obs.ndim == 3
        assert state.shape == (1,3,STATE_SHAPE)
        assert obs.shape == (1,3,OBS_SHAPE)
        assert env.players['SB'].stack == 4
        assert env.players['BB'].stack == 5
        assert env.players['SB'].street_total == 1
        assert env.players['BB'].street_total == 0
        assert env.pot == 2
        assert state[:,-1][:,env.state_mapping['street']] == self.env_params['starting_street']
        assert state[:,-1][:,env.state_mapping['hero_position']] == pdt.Position.BB
        assert state[:,-1][:,env.state_mapping['last_position']] == pdt.Position.SB
        assert state[:,-1][:,env.state_mapping['last_action']] == pdt.Action.BET
        assert state[:,-1][:,env.state_mapping['last_betsize']] == 1
        assert state[:,-1][:,env.state_mapping['hero_stacksize']] == self.env_params['stacksize']
        assert state[:,-1][:,env.state_mapping['player2_stacksize']] == self.env_params['stacksize'] - 1
        assert state[:,-1][:,env.state_mapping['amount_to_call']] == 1
        self.assertAlmostEqual(state[:,-1][:,env.state_mapping['pot_odds']][0],0.333,places=2)
        assert done == False
        assert np.array_equal(mask,np.array([0., 1., 1., 0., 1.]))
        assert np.array_equal(betsize_mask,np.array([1.,1.]))

        state,obs,done,mask,betsize_mask = env.step(ACTION_FOLD)
        assert done == True
        assert env.players['SB'].stack == 6
        assert env.players['BB'].stack == 5
        assert env.players['BB'].status == Status.FOLDED

    def testBetRaiseCall(self):
        params = copy.deepcopy(self.env_params)
        params['starting_street'] = pdt.Street.RIVER
        params['stacksize'] = 5
        params['pot'] = 1
        env = Poker(params)
        state,obs,done,mask,betsize_mask = env.reset()
        state,obs,done,mask,betsize_mask = env.step(ACTION_BET)
        assert state.ndim == 3
        assert obs.ndim == 3
        assert state.shape == (1,2,STATE_SHAPE)
        assert obs.shape == (1,2,OBS_SHAPE)
        assert env.players['SB'].stack == 5
        assert env.players['BB'].stack == 4
        assert env.players['SB'].street_total == 0
        assert env.players['BB'].street_total == 1
        assert env.pot == 2
        assert state[:,-1][:,env.state_mapping['street']] == pdt.Street.RIVER
        assert state[:,-1][:,env.state_mapping['hero_position']] == pdt.Position.SB
        assert state[:,-1][:,env.state_mapping['last_position']] == pdt.Position.BB
        assert state[:,-1][:,env.state_mapping['last_action']] == pdt.Action.BET
        assert state[:,-1][:,env.state_mapping['last_betsize']] == 1
        assert state[:,-1][:,env.state_mapping['hero_stacksize']] == 5
        assert state[:,-1][:,env.state_mapping['player2_stacksize']] == 5 - 1
        assert state[:,-1][:,env.state_mapping['amount_to_call']] == 1
        self.assertAlmostEqual(state[:,-1][:,env.state_mapping['pot_odds']][0],0.333,places=2)
        assert done == False
        assert np.array_equal(mask,np.array([0., 1., 1., 0., 1.]))
        assert np.array_equal(betsize_mask,np.array([1.,1.]))

        state,obs,done,mask,betsize_mask = env.step(ACTION_RAISE)
        assert state.ndim == 3
        assert obs.ndim == 3
        assert state.shape == (1,3,STATE_SHAPE)
        assert obs.shape == (1,3,OBS_SHAPE)
        assert env.players['SB'].stack == 1
        assert env.players['BB'].stack == 4
        assert env.players['SB'].street_total == 4
        assert env.players['BB'].street_total == 1
        assert env.pot == 6
        assert state[:,-1][:,env.state_mapping['street']] == self.env_params['starting_street']
        assert state[:,-1][:,env.state_mapping['hero_position']] == pdt.Position.BB
        assert state[:,-1][:,env.state_mapping['last_position']] == pdt.Position.SB
        assert state[:,-1][:,env.state_mapping['last_action']] == pdt.Action.RAISE
        assert state[:,-1][:,env.state_mapping['last_betsize']] == 4
        assert state[:,-1][:,env.state_mapping['hero_stacksize']] == self.env_params['stacksize'] - 1
        assert state[:,-1][:,env.state_mapping['player2_stacksize']] == self.env_params['stacksize'] - 4
        assert state[:,-1][:,env.state_mapping['amount_to_call']] == 3
        self.assertAlmostEqual(state[:,-1][:,env.state_mapping['pot_odds']][0],0.33,places=2)
        assert done == False
        assert np.array_equal(mask,np.array([0., 1., 1., 0., 1.]))
        assert np.array_equal(betsize_mask,np.array([1.,0.]))

        state,obs,done,mask,betsize_mask = env.step(ACTION_CALL)
        assert done == True
        assert env.players['SB'].stack == 10
        assert env.players['BB'].stack == 1

    def testBetRestrictions(self):
        env = Poker(self.env_params)
        state,obs,done,mask,betsize_mask = env.reset()
        state,obs,done,mask,betsize_mask = env.step(ACTION_BET)
        state,obs,done,mask,betsize_mask = env.step(ACTION_RAISE)
        state,obs,done,mask,betsize_mask = env.step(ACTION_MIN_RAISE)
        assert env.players['SB'].stack == 1
        assert env.players['SB'].status == Status.ACTIVE
        assert env.players['BB'].stack == 0
        assert env.players['BB'].status == Status.ALLIN
        assert state[0,-1,env.state_mapping['blind']] == pdt.Blind.NO_BLIND
        assert np.array_equal(mask,np.array([0., 1., 1., 0., 0.]))
        assert np.array_equal(betsize_mask,np.array([0.,0.]))
        state,obs,done,mask,betsize_mask = env.step(ACTION_CALL)
        assert done == True
        assert env.players['SB'].stack == 11
        assert env.players['BB'].stack == 0

    def testTies(self):
        env = Poker(self.env_params)
        state,obs,done,mask,betsize_mask = env.reset()
        # Modify board and hands
        env.board = [14,0,13,1,12,2,2,2,2,3]
        env.players['SB'].hand = [[11,3],[10,3],[3,2],[3,3]]
        env.players['BB'].hand = [[11,2],[10,2],[4,0],[4,3]]
        state,obs,done,mask,betsize_mask = env.step(ACTION_BET)
        state,obs,done,mask,betsize_mask = env.step(ACTION_CALL)
        assert done == True
        assert env.players['SB'].stack == 5.5
        assert env.players['BB'].stack == 5.5

    def testBlindInitialization(self):
        params = copy.deepcopy(self.env_params)
        params['starting_street'] = pdt.Street.PREFLOP
        params['pot'] = 0
        env = Poker(params)
        state,obs,done,mask,betsize_mask = env.reset()
        assert env.players['SB'].stack == 4.5
        assert env.players['BB'].stack == 4.
        assert env.players['SB'].street_total == 0.5
        assert env.players['BB'].street_total == 1.
        assert state[0,-1,env.state_mapping['blind']] == pdt.Blind.POSTED
        assert state[:,-1][:,env.state_mapping['hero_position']] == pdt.Position.SB
        assert state[:,-1][:,env.state_mapping['last_position']] == pdt.Position.BB
        assert done == False

    def testStreetIncrement(self):
        params = copy.deepcopy(self.env_params)
        params['starting_street'] = pdt.Street.TURN
        params['pot'] = 1
        env = Poker(params)
        state,obs,done,mask,betsize_mask = env.reset()
        assert env.board[-2] == 0
        assert env.board[-1] == 0
        state,obs,done,mask,betsize_mask = env.step(ACTION_BET)
        state,obs,done,mask,betsize_mask = env.step(ACTION_CALL)
        assert env.street == pdt.Street.RIVER
        assert env.board[-2] != 0
        state,obs,done,mask,betsize_mask = env.step(ACTION_BET)
        state,obs,done,mask,betsize_mask = env.step(ACTION_CALL)
        assert done == True
        del env
        params['starting_street'] = pdt.Street.PREFLOP
        params['pot'] = 0
        env = Poker(params)
        state,obs,done,mask,betsize_mask = env.reset()
        state,obs,done,mask,betsize_mask = env.step(ACTION_CALL)
        assert state[:,-1][:,env.state_mapping['hero_position']] == pdt.Position.BB
        assert state[:,-1][:,env.state_mapping['last_position']] == pdt.Position.SB
        assert env.pot == 2
        state,obs,done,mask,betsize_mask = env.step(ACTION_RAISE)
        assert env.players['BB'].stack == 2
        assert env.players['SB'].stack == 4
        assert env.pot == 4
        assert state[:,-1][:,env.state_mapping['hero_position']] == pdt.Position.SB
        assert state[:,-1][:,env.state_mapping['last_position']] == pdt.Position.BB
        state,obs,done,mask,betsize_mask = env.step(ACTION_CALL)
        assert state[:,-1][:,env.state_mapping['hero_position']] == pdt.Position.BB
        assert state[:,-1][:,env.state_mapping['player2_position']] == pdt.Position.SB
        assert state[:,-1][:,env.state_mapping['last_position']] == pdt.Position.BTN
        assert state[:,-1][:,env.state_mapping['last_aggressive_position']] == pdt.Position.BB
        assert env.street == pdt.Street.FLOP
        state,obs,done,mask,betsize_mask = env.step(ACTION_CHECK)
        state,obs,done,mask,betsize_mask = env.step(ACTION_CHECK)
        assert env.street == pdt.Street.TURN
        state,obs,done,mask,betsize_mask = env.step(ACTION_CHECK)
        state,obs,done,mask,betsize_mask = env.step(ACTION_CHECK)
        assert env.street == pdt.Street.RIVER
        state,obs,done,mask,betsize_mask = env.step(ACTION_CHECK)
        state,obs,done,mask,betsize_mask = env.step(ACTION_CHECK)
        assert done == True

    def testThreePlayers(self):
        params = copy.deepcopy(self.env_params)
        params['n_players'] = 3
        params['starting_street'] = pdt.Street.PREFLOP
        params['pot'] = 0
        env = Poker(params)
        state,obs,done,mask,betsize_mask = env.reset()
        assert state[:,-1][:,env.state_mapping['hero_position']] == pdt.Position.BTN
        assert state[:,-1][:,env.state_mapping['player1_position']] == pdt.Position.BTN
        assert state[:,-1][:,env.state_mapping['player2_position']] == pdt.Position.SB
        assert state[:,-1][:,env.state_mapping['player3_position']] == pdt.Position.BB
        assert env.street == pdt.Street.PREFLOP
        assert env.players.num_active_players == 3
        state,obs,done,mask,betsize_mask = env.step(ACTION_RAISE)
        assert env.players['SB'].stack == 4.5
        assert env.players['BB'].stack == 4.
        assert env.players['BTN'].stack == 1.5
        assert env.players['SB'].street_total == 0.5
        assert env.players['BB'].street_total == 1.
        assert env.players['BTN'].street_total == 3.5
        state,obs,done,mask,betsize_mask = env.step(ACTION_FOLD)
        assert env.players['SB'].status == Status.FOLDED
        assert env.players['BB'].status == Status.ACTIVE
        assert env.players['BTN'].status == Status.ACTIVE
        state,obs,done,mask,betsize_mask = env.step(ACTION_CALL)
        assert env.players['SB'].stack == 4.5
        assert env.players['BB'].stack == 1.5
        assert env.players['BTN'].stack == 1.5
        assert env.players['SB'].street_total == 0.
        assert env.players['BB'].street_total == 0.
        assert env.players['BTN'].street_total == 0.
        assert state[:,-1][:,env.state_mapping['pot']] == 7.5
        assert env.pot == 7.5
        assert env.street == pdt.Street.FLOP
        assert env.players.num_active_players == 2
        assert state[:,-1][:,env.state_mapping['hero_position']] == pdt.Position.BB
        state,obs,done,mask,betsize_mask = env.step(ACTION_CHECK)
        state,obs,done,mask,betsize_mask = env.step(ACTION_CHECK)
        assert env.street == pdt.Street.TURN
        state,obs,done,mask,betsize_mask = env.step(ACTION_CHECK)
        state,obs,done,mask,betsize_mask = env.step(ACTION_CHECK)
        assert env.street == pdt.Street.RIVER
        state,obs,done,mask,betsize_mask = env.step(ACTION_CHECK)
        state,obs,done,mask,betsize_mask = env.step(ACTION_CHECK)
        assert done == True
        assert env.players['SB'].stack == 4.5
        assert env.players['BB'].stack == 9
        assert env.players['BTN'].stack == 1.5
        del env
        params['n_players'] = 3
        params['starting_street'] = pdt.Street.PREFLOP
        params['pot'] = 0
        env = Poker(params)
        state,obs,done,mask,betsize_mask = env.reset()
        assert state[:,-1][:,env.state_mapping['hero_position']] == pdt.Position.BTN
        state,obs,done,mask,betsize_mask = env.step(ACTION_CALL)
        assert state[:,-1][:,env.state_mapping['hero_position']] == pdt.Position.SB
        assert env.players['SB'].street_total == 0.5
        state,obs,done,mask,betsize_mask = env.step(ACTION_CALL)
        assert env.players['SB'].street_total == 1
        assert state[:,-1][:,env.state_mapping['hero_position']] == pdt.Position.BB
        state,obs,done,mask,betsize_mask = env.step(ACTION_CHECK)
        assert env.street == pdt.Street.FLOP
        assert env.pot == 3
        state,obs,done,mask,betsize_mask = env.step(ACTION_CHECK)
        state,obs,done,mask,betsize_mask = env.step(ACTION_CHECK)
        state,obs,done,mask,betsize_mask = env.step(ACTION_CHECK)
        assert env.street == pdt.Street.TURN
        state,obs,done,mask,betsize_mask = env.step(ACTION_CHECK)
        state,obs,done,mask,betsize_mask = env.step(ACTION_CHECK)
        state,obs,done,mask,betsize_mask = env.step(ACTION_CHECK)
        assert env.street == pdt.Street.RIVER
        state,obs,done,mask,betsize_mask = env.step(ACTION_CHECK)
        state,obs,done,mask,betsize_mask = env.step(ACTION_CHECK)
        state,obs,done,mask,betsize_mask = env.step(ACTION_CHECK)
        assert done == True
        assert env.players['SB'].stack == 7
        assert env.players['BB'].stack == 4
        assert env.players['BTN'].stack == 4.
        del env
        params['n_players'] = 3
        params['starting_street'] = pdt.Street.PREFLOP
        params['pot'] = 0
        env = Poker(params)
        state,obs,done,mask,betsize_mask = env.reset()
        state,obs,done,mask,betsize_mask = env.step(ACTION_RAISE)
        state,obs,done,mask,betsize_mask = env.step(ACTION_FOLD)
        state,obs,done,mask,betsize_mask = env.step(ACTION_FOLD)
        assert done == True
        assert env.players['SB'].stack == 4.5
        assert env.players['BB'].stack == 4
        assert env.players['BTN'].stack == 6.5
        
    def testBetLimits(self):
        params = copy.deepcopy(self.env_params)
        # Limit
        params['bet_type'] = pdt.LimitTypes.LIMIT
        params['n_players'] = 3
        params['starting_street'] = pdt.Street.PREFLOP
        params['pot'] = 0
        env = Poker(params)
        state,obs,done,mask,betsize_mask = env.reset()
        assert state[:,-1][:,env.state_mapping['pot']] == 1.5
        assert state[:,-1][:,env.state_mapping['hero_position']] == pdt.Position.BTN
        state,obs,done,mask,betsize_mask = env.step(ACTION_RAISE)
        assert state[:,-1][:,env.state_mapping['pot']] == 3.5
        assert env.players['BTN'].stack == 3
        assert env.players['BB'].stack == 4
        assert env.players['SB'].stack == 4.5
        assert env.players['SB'].street_total == 0.5
        assert state[:,-1][:,env.state_mapping['hero_position']] == pdt.Position.SB
        assert state[:,-1][:,env.state_mapping['last_aggressive_betsize']] == 2
        assert env.street == pdt.Street.PREFLOP
        state,obs,done,mask,betsize_mask = env.step(ACTION_RAISE)
        assert env.players['BTN'].stack == 3
        assert env.players['BB'].stack == 4
        assert env.players['SB'].stack == 2
        assert env.players['SB'].street_total == 3.
        assert state[:,-1][:,env.state_mapping['hero_position']] == pdt.Position.BB
        assert state[:,-1][:,env.state_mapping['last_aggressive_betsize']] == 2.5
        assert state[:,-1][:,env.state_mapping['pot']] == 6
        assert env.street == pdt.Street.PREFLOP
        state,obs,done,mask,betsize_mask = env.step(ACTION_CALL)
        assert state[:,-1][:,env.state_mapping['pot']] == 8
        assert state[:,-1][:,env.state_mapping['hero_position']] == pdt.Position.BTN
        assert env.street == pdt.Street.PREFLOP
        state,obs,done,mask,betsize_mask = env.step(ACTION_CALL)
        assert state[:,-1][:,env.state_mapping['pot']] == 9
        assert state[:,-1][:,env.state_mapping['hero_position']] == pdt.Position.SB
        assert env.street == pdt.Street.FLOP
        del env
        params['bet_type'] = pdt.LimitTypes.POT_LIMIT
        params['n_players'] = 3
        params['starting_street'] = pdt.Street.PREFLOP
        params['pot'] = 0
        params['stacksize'] = 100
        env = Poker(params)
        state,obs,done,mask,betsize_mask = env.reset()
        assert state[:,-1][:,env.state_mapping['hero_position']] == pdt.Position.BTN
        state,obs,done,mask,betsize_mask = env.step(ACTION_RAISE)
        assert state[:,-1][:,env.state_mapping['hero_position']] == pdt.Position.SB
        assert env.players['BTN'].stack == 96.5
        assert env.players['BTN'].street_total == 3.5
        state,obs,done,mask,betsize_mask = env.step(ACTION_RAISE)
        assert env.players['SB'].stack == 88.5
        assert env.players['SB'].street_total == 11.5
        assert state[:,-1][:,env.state_mapping['last_aggressive_betsize']] == 11
        assert state[:,-1][:,env.state_mapping['pot']] == 16
        state,obs,done,mask,betsize_mask = env.step(ACTION_FOLD)
        state,obs,done,mask,betsize_mask = env.step(ACTION_RAISE)
        assert env.players['BTN'].stack == 64.5
        assert env.players['BTN'].street_total == 35.5
        assert state[:,-1][:,env.state_mapping['last_aggressive_betsize']] == 32
        assert state[:,-1][:,env.state_mapping['pot']] == 48
        del env
        params['bet_type'] = pdt.LimitTypes.POT_LIMIT
        params['n_players'] = 3
        params['starting_street'] = pdt.Street.PREFLOP
        params['pot'] = 0
        params['stacksize'] = 100
        env = Poker(params)
        state,obs,done,mask,betsize_mask = env.reset()
        state,obs,done,mask,betsize_mask = env.step(ACTION_CALL)
        state,obs,done,mask,betsize_mask = env.step(ACTION_RAISE)
        assert env.players['SB'].stack == 96
        assert env.players['SB'].street_total == 4
        state,obs,done,mask,betsize_mask = env.step(ACTION_CALL)
        state,obs,done,mask,betsize_mask = env.step(ACTION_CALL)
        assert env.street == pdt.Street.FLOP
        assert state[:,-1][:,env.state_mapping['pot']] == 12
        state,obs,done,mask,betsize_mask = env.step(ACTION_BET)
        assert state[:,-1][:,env.state_mapping['pot']] == 24
        assert env.players['SB'].stack == 84
        assert env.players['SB'].street_total == 12
        state,obs,done,mask,betsize_mask = env.step(ACTION_RAISE)
        assert env.players['BB'].stack == 48
        assert env.players['BB'].street_total == 48
        assert state[:,-1][:,env.state_mapping['pot']] == 72
        #TODO No Limit
        # params['bet_limit'] = pdt.LimitTypes.NO_LIMIT
        # params['n_players'] = 3
        # params['starting_street'] = pdt.Street.PREFLOP
        # params['pot'] = 0
        # env = Poker(params)
        # state,obs,done,mask,betsize_mask = env.reset()
        # del env
        # Pot Limit
        # Test reraise, preflop raise, sb raise preflop. raise vs bet

    def testAllin(self):
        params = copy.deepcopy(self.env_params)
        params['n_players'] = 3
        params['starting_street'] = pdt.Street.PREFLOP
        params['pot'] = 0
        env = Poker(params)
        state,obs,done,mask,betsize_mask = env.reset()
        state,obs,done,mask,betsize_mask = env.step(ACTION_RAISE)
        assert env.players['BTN'].stack == 1.5
        assert env.players['BTN'].street_total == 3.5
        state,obs,done,mask,betsize_mask = env.step(ACTION_FOLD)
        assert env.players['SB'].status == Status.FOLDED
        state,obs,done,mask,betsize_mask = env.step(ACTION_RAISE)
        assert env.players['BB'].stack == 0
        assert env.players['BB'].street_total == 5
        assert env.players['BB'].status == Status.ALLIN
        state,obs,done,mask,betsize_mask = env.step(ACTION_CALL)
        assert env.players['BB'].stack == 10.5
        assert env.players['SB'].stack == 4.5
        assert env.players['BTN'].stack == 0
        assert env.players['BTN'].street_total == 0
        assert env.street == pdt.Street.RIVER
        assert done == True

    def testActor(self):
        params = copy.deepcopy(self.env_params)
        env = Poker(params)
        nA = env.action_space
        nB = env.betsize_space
        nS = env.state_space
        seed = 152
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        params['device'] = device
        params['maxlen'] = 10
        params['embedding_size'] = 128
        actor = OmahaActor(seed,nS,nA,nB,params)
        state,obs,done,mask,betsize_mask = env.reset()
        output = actor(state,mask,betsize_mask)
        state,obs,done,mask,betsize_mask = env.step(ACTION_BET)
        output = actor(state,mask,betsize_mask)
        assert isinstance(output['action_probs'],torch.Tensor)
        assert isinstance(output['action_prob'],torch.Tensor)

    def testCritic(self):
        params = copy.deepcopy(self.env_params)
        env = Poker(params)
        nA = env.action_space
        nB = env.betsize_space
        nS = env.state_space
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seed = 152
        params['maxlen'] = 10
        params['embedding_size'] = 128
        params['transformer_in'] = 256
        params['transformer_out'] = 128
        params['transformer_out'] = 128
        params['device'] = device
        critic = OmahaObsQCritic(seed,nS,nA,nB,params)
        state,obs,done,mask,betsize_mask = env.reset()
        output = critic(obs)
        assert isinstance(output['value'],torch.Tensor)

    # def testCombined(self):
    #     params = copy.deepcopy(self.env_params)
    #     env = Poker(params)
    #     nA = env.action_space
    #     nB = env.betsize_space
    #     nS = env.state_space
    #     seed = 152
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     params['device'] = device
    #     params['maxlen'] = 10
    #     params['embedding_size'] = 128
    #     params['transformer_in'] = 7718
    #     params['transformer_out'] = 128
    #     net = CombinedNet(seed,nS,nA,nB,params)
    #     state,obs,done,mask,betsize_mask = env.reset()
    #     output = net(state,mask,betsize_mask)
    #     assert isinstance(output['value'],torch.Tensor)

    def testMasks(self):
        params = copy.deepcopy(self.env_params)
        params['stacksize'] = 5
        params['n_players'] = 2
        params['starting_street'] = pdt.Street.PREFLOP
        params['pot'] = 0
        env = Poker(params)
        state,obs,done,mask,betsize_mask = env.reset()
        assert state[:,-1][:,env.state_mapping['pot']] == 1.5
        assert state[:,-1][:,env.state_mapping['player1_stacksize']] == 4.5
        assert state[:,-1][:,env.state_mapping['player1_position']] == pdt.Position.SB
        assert state[:,-1][:,env.state_mapping['player2_stacksize']] == 4
        assert state[:,-1][:,env.state_mapping['player2_position']] == pdt.Position.BB
        assert state[:,-1][:,env.state_mapping['street']] == pdt.Street.PREFLOP
        assert env.current_player == 'SB'
        assert np.array_equal(betsize_mask,np.array([1,1]))
        assert np.array_equal(mask,np.array([0,1,1,0,1]))
        state,obs,done,mask,betsize_mask = env.step(ACTION_RAISE)
        assert env.current_player == 'BB'
        assert state[:,-1][:,env.state_mapping['pot']] == 4
        assert state[:,-1][:,env.state_mapping['player1_stacksize']] == 4
        assert state[:,-1][:,env.state_mapping['player1_position']] == pdt.Position.BB
        assert state[:,-1][:,env.state_mapping['player2_stacksize']] == 2
        assert state[:,-1][:,env.state_mapping['player2_position']] == pdt.Position.SB
        assert state[:,-1][:,env.state_mapping['street']] == pdt.Street.PREFLOP
        assert np.array_equal(mask,np.array([0,1,1,0,1]))
        assert np.array_equal(betsize_mask,np.array([1,0]))
        state,obs,done,mask,betsize_mask = env.step(ACTION_RAISE)
        assert state[:,-1][:,env.state_mapping['pot']] == 8
        assert state[:,-1][:,env.state_mapping['player1_stacksize']] == 2
        assert state[:,-1][:,env.state_mapping['player1_position']] == pdt.Position.SB
        assert state[:,-1][:,env.state_mapping['player2_stacksize']] == 0
        assert state[:,-1][:,env.state_mapping['player2_position']] == pdt.Position.BB
        assert state[:,-1][:,env.state_mapping['street']] == pdt.Street.PREFLOP
        assert np.array_equal(mask,np.array([0,1,1,0,0]))
        assert np.array_equal(betsize_mask,np.array([0,0]))

    def testEnvCategoryMapping(self):
        params = copy.deepcopy(self.env_params)
        params['stacksize'] = 50
        params['n_players'] = 2
        params['starting_street'] = pdt.Street.PREFLOP
        params['pot'] = 0
        env = Poker(params)
        state,obs,done,mask,betsize_mask = env.reset()
        assert env.convert_to_category(pdt.NetworkActions.RAISE,3)[0] == 4
        assert env.convert_to_category(pdt.NetworkActions.RAISE,2)[0] == 3
        assert env.convert_to_category(pdt.NetworkActions.CALL,0.5)[0] == 2
        assert env.convert_to_category(pdt.NetworkActions.CHECK,0)[0] == 0
        state,obs,done,mask,betsize_mask = env.step(ACTION_RAISE)
        assert env.convert_to_category(pdt.NetworkActions.RAISE,9)[0] == 4
        assert env.convert_to_category(pdt.NetworkActions.RAISE,5)[0] == 3
        assert env.convert_to_category(pdt.NetworkActions.CALL,2)[0] == 2
        assert env.convert_to_category(pdt.NetworkActions.CHECK,0)[0] == 0
        state,obs,done,mask,betsize_mask = env.step(ACTION_CALL)
        assert env.convert_to_category(pdt.NetworkActions.BET,6)[0] == 4
        assert env.convert_to_category(pdt.NetworkActions.BET,3)[0] == 3
        assert env.convert_to_category(pdt.NetworkActions.FOLD,0)[0] == 1
        state,obs,done,mask,betsize_mask = env.step(ACTION_BET)
        assert env.convert_to_category(pdt.NetworkActions.RAISE,24)[0] == 4
        assert env.convert_to_category(pdt.NetworkActions.RAISE,12)[0] == 3
        assert env.convert_to_category(pdt.NetworkActions.CALL,6)[0] == 2
        assert env.convert_to_category(pdt.NetworkActions.FOLD,0)[0] == 1
        state,obs,done,mask,betsize_mask = env.step(ACTION_RAISE)
        assert env.convert_to_category(pdt.NetworkActions.RAISE,47)[0] == 4
        assert env.convert_to_category(pdt.NetworkActions.RAISE,42)[0] == 3
        assert env.convert_to_category(pdt.NetworkActions.CALL,18)[0] == 2
        assert env.convert_to_category(pdt.NetworkActions.FOLD,0)[0] == 1
        del env
        params['stacksize'] = 3
        params['n_players'] = 2
        params['starting_street'] = pdt.Street.PREFLOP
        params['pot'] = 0
        env = Poker(params)
        state,obs,done,mask,betsize_mask = env.reset()
        assert env.convert_to_category(pdt.NetworkActions.RAISE,3)[0] == 4
        print('check',env.convert_to_category(pdt.NetworkActions.RAISE,2)[0])
        assert env.convert_to_category(pdt.NetworkActions.RAISE,2)[0] == 3
        assert env.convert_to_category(pdt.NetworkActions.CALL,0)[0] == 2
        assert env.convert_to_category(pdt.NetworkActions.FOLD,0)[0] == 1
        
    def testStreetInitialization(self):
        params = copy.deepcopy(self.env_params)
        params['stacksize'] = 50
        params['n_players'] = 2
        params['starting_street'] = pdt.Street.RIVER
        params['pot'] = 1
        env = Poker(params)
        state,obs,done,mask,betsize_mask = env.reset()
        assert state[:,-1][:,env.state_mapping['player1_position']] == pdt.Position.BB
        assert state[:,-1][:,env.state_mapping['hero_position']] == pdt.Position.BB

    def additionalTests(self):
        params = copy.deepcopy(self.env_params)
        params['stacksize'] = 5
        params['n_players'] = 2
        params['starting_street'] = pdt.Street.PREFLOP
        params['pot'] = 0
        env = Poker(params)
        state,obs,done,mask,betsize_mask = env.reset()
        state,obs,done,mask,betsize_mask = env.step(ACTION_RAISE)
        assert state[:,-1][:,env.state_mapping['player2_stacksize']] == 2
        state,obs,done,mask,betsize_mask = env.step(ACTION_CALL)
        assert state[:,-1][:,env.state_mapping['player1_stacksize']] == 2
        assert state[:,-1][:,env.state_mapping['street']] == 1
        state,obs,done,mask,betsize_mask = env.step(ACTION_BET)
        state,obs,done,mask,betsize_mask = env.step(ACTION_CALL)
        assert state[:,-1][:,env.state_mapping['player1_stacksize']] == 0
        assert state[:,-1][:,env.state_mapping['player2_stacksize']] == 0
        assert state[:,-1][:,env.state_mapping['street']] == 3
        assert done == True

    def preflopTests(self):
        """Facing sb call. Sb min raise."""
        params = copy.deepcopy(self.env_params)
        params['stacksize'] = 5
        params['n_players'] = 2
        params['starting_street'] = pdt.Street.PREFLOP
        params['pot'] = 0
        env = Poker(params)
        state,obs,done,mask,betsize_mask = env.reset()
        assert np.array_equal(mask,np.array([0,1,1,0,1]))
        state,obs,done,mask,betsize_mask = env.step(ACTION_CALL)
        assert state[:,-1][:,env.state_mapping['player2_stacksize']] == 4
        assert state[:,-1][:,env.state_mapping['player1_stacksize']] == 4
        assert np.array_equal(mask,np.array([1,0,0,0,1]))
        del env
        env = Poker(params)
        state,obs,done,mask,betsize_mask = env.reset()
        state,obs,done,mask,betsize_mask = env.step(ACTION_MIN_RAISE)
        assert state[:,-1][:,env.state_mapping['player2_stacksize']] == 3
        assert state[:,-1][:,env.state_mapping['player1_stacksize']] == 4

    def betsizingTests(self):
        params = copy.deepcopy(self.env_params)
        params['stacksize'] = 5
        params['n_players'] = 2
        params['starting_street'] = pdt.Street.PREFLOP
        params['pot'] = 0
        env = Poker(params)
        state,obs,done,mask,betsize_mask = env.reset()
        betsize = env.return_potlimit_betsize(action=4,betsize_category=0)
        assert betsize == 1.5
        betsize = env.return_potlimit_betsize(action=4,betsize_category=1)
        assert betsize == 2.5
        betsize = env.return_potlimit_betsize(action=2,betsize_category=0)
        assert betsize == 0.5
        state,obs,done,mask,betsize_mask = env.step(ACTION_CALL)
        betsize = env.return_potlimit_betsize(action=4,betsize_category=0)
        assert betsize == 1
        betsize = env.return_potlimit_betsize(action=4,betsize_category=1)
        assert betsize == 2

    # def testOutcome(self):
    #     params = self.env_params
    #     params['stacksize'] = 5
    #     params['starting_street'] = pdt.Street.TURN
    #     params['pot'] = 1
    #     env = Poker(self.env_params)
    #     state,obs,done,mask,betsize_mask = env.reset()
    #     env.players['SB'].hand = [[7, 1], [5, 3], [14, 2], [10, 2]]
    #     env.players['BB'].hand = [[14, 3], [2, 1], [2, 4], [11, 1]]
    #     env.board = [[10, 3], [2, 2], [4, 3], [13, 3], [4, 2]]
    #     state,obs,done,mask,betsize_mask = env.step(ACTION_BET)
    #     state,obs,done,mask,betsize_mask = env.step(ACTION_CALL)
    #     assert state[:,-1][:,env.state_mapping['player2_stacksize']] == 7
    #     assert state[:,-1][:,env.state_mapping['player1_stacksize']] == 4
    #     assert done == True

def envTestSuite():
    suite = unittest.TestSuite()
    suite.addTest(TestEnv('testInitialization'))
    suite.addTest(TestEnv('testReset'))
    suite.addTest(TestEnv('testCheckCheck'))
    suite.addTest(TestEnv('testCheckBetFold'))
    suite.addTest(TestEnv('testBetRaiseCall'))
    suite.addTest(TestEnv('testBetRestrictions'))
    suite.addTest(TestEnv('testTies'))
    suite.addTest(TestEnv('testBlindInitialization'))
    suite.addTest(TestEnv('testStreetIncrement'))
    suite.addTest(TestEnv('testThreePlayers'))
    suite.addTest(TestEnv('testBetLimits'))
    suite.addTest(TestEnv('testAllin'))
    suite.addTest(TestEnv('testActor'))
    suite.addTest(TestEnv('testCritic'))
    # suite.addTest(TestEnv('testCombined'))
    suite.addTest(TestEnv('testMasks'))
    suite.addTest(TestEnv('testEnvCategoryMapping'))
    suite.addTest(TestEnv('testStreetInitialization'))
    suite.addTest(TestEnv('additionalTests'))
    suite.addTest(TestEnv('testPreflop'))
    suite.addTest(TestEnv('testBetsizing'))
    suite.addTest(TestEnv('testOutcome'))
    return suite

if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(envTestSuite())