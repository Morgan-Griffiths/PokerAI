from poker.env import Poker,Status
import poker.datatypes as pdt
from poker.config import Config
import numpy as np

import torch
import unittest
import numpy as np
import os
import copy

from poker.config import Config
from poker.data_classes import Card,Evaluator
import poker.datatypes as pdt
from utils.cardlib import winner,holdem_winner,encode

ACTION_CHECK = {
    'action':torch.Tensor([0]).long(),
    'action_category':torch.Tensor([0]).long(),
    'action_probs':torch.zeros(5).fill_(0.2),
    'action_prob':torch.Tensor([0.2]),
    'betsize':torch.Tensor([0])
}
ACTION_FOLD = {
    'action':torch.Tensor([1]).long(),
    'action_category':torch.Tensor([1]).long(),
    'action_probs':torch.zeros(5).fill_(0.2),
    'action_prob':torch.Tensor([0.2]),
    'betsize':torch.Tensor([0])
}
ACTION_BET = {
    'action':torch.Tensor([4]).long(),
    'action_category':torch.Tensor([3]).long(),
    'action_probs':torch.zeros(5).fill_(0.2),
    'action_prob':torch.Tensor([0.2]),
    'betsize':torch.Tensor([1])
}
ACTION_CALL = {
    'action':torch.Tensor([2]).long(),
    'action_category':torch.Tensor([2]).long(),
    'action_probs':torch.zeros(5).fill_(0.2),
    'action_prob':torch.Tensor([0.2]),
    'betsize':torch.Tensor([0])
}
ACTION_RAISE = {
    'action':torch.Tensor([4]).long(),
    'action_category':torch.Tensor([4]).long(),
    'action_probs':torch.zeros(5).fill_(0.2),
    'action_prob':torch.Tensor([0.2]),
    'betsize':torch.Tensor([1])
}
ACTION_MIN_RAISE = {
    'action':torch.Tensor([4]).long(),
    'action_category':torch.Tensor([4]).long(),
    'action_probs':torch.zeros(5).fill_(0.2),
    'action_prob':torch.Tensor([0.2]),
    'betsize':torch.Tensor([0])
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
        global_mapping = {
            'board':np.array([0,1,2,3,4,5,6,7,8,9]),
            'board_ranks':np.array([0,2,4,6,8]),
            'board_suits':np.array([1,3,5,7,9]),
            'street':np.array([10]),
            'last_aggressive_position':np.array([11]),
            'last_aggressive_action':np.array([12]),
            'last_aggressive_betsize':np.array([13]),
            'last_position':np.array([14]),
            'last_action':np.array([15]),
            'last_betsize':np.array([16]),
            'blind':np.array([17]),
            'pot':np.array([18]),
            'amount_to_call':np.array([19]),
            'pot_odds':np.array([20]),
            'p1_position':np.array([21]),
            'p1_stacksize':np.array([22]),
            'p1_street_total':np.array([23]),
            'p1_status':np.array([24]),
            'p2_position':np.array([25]),
            'p2_stacksize':np.array([26]),
            'p2_street_total':np.array([27]),
            'p2_status':np.array([28]),
            'p3_position':np.array([29]),
            'p3_stacksize':np.array([30]),
            'p3_street_total':np.array([31]),
            'p3_status':np.array([32]),
        }
        state_mapping = {
            'hero_position':0,
            'hero_stacksize':1,
            'hero_hand':[2,3,4,5,6,7,8,9],
            'hero_ranks':[2,4,6,8],
            'hero_suits':[3,5,7,9],
            'board':[10,11,12,13,14,15,16,17,18,19],
            'board_ranks':[10,12,14,16,18],
            'board_suits':[11,13,15,17,19],
            'street':20,
            'last_aggressive_position':21,
            'last_aggressive_action':22,
            'last_aggressive_betsize':23,
            'last_position':24,
            'last_action':25,
            'last_betsize':26,
            'blind':27,
            'pot':28,
            'amount_to_call':29,
            'pot_odds':30,
            'player1_position':31,
            'player1_stacksize':32,
            'player1_street_total':33,
            'player1_status':34,
            'player2_position':35,
            'player2_stacksize':36,
            'player2_street_total':37,
            'player2_status':38,
            'player3_position':39,
            'player3_stacksize':40,
            'player3_street_total':41,
            'player3_status':42,
            'ordinal': [0,20,21,22,23,26,27,30,31,34],
            'continuous': [1,23,25,26,28,29,32,33],
            'hand_board':[2,3,4,5,6,7,8,9] + [10,11,12,13,14,15,16,17,18,19]
        }
        obs_mapping = {
            'hero_position':0,
            'hero_stacksize':1,
            'hero_hand':[2,3,4,5,6,7,8,9],
            'hero_ranks':[2,4,6,8],
            'hero_suits':[3,5,7,9],
            'villain_position':10,
            'villain_stacksize':11,
            'villain_hand':[12,13,14,15,16,17,18,19],
            'villain_ranks':[12,14,16,18],
            'villain_suits':[13,15,17,19],
            'board':[20,21,22,23,24,25,26,27,28,29],
            'board_ranks':[20,22,24,26,28],
            'board_suits':[21,23,25,27,29],
            'street':30,
            'last_aggressive_position':31,
            'last_aggressive_action':32,
            'last_aggressive_betsize':33,
            'last_position':34,
            'last_action':35,
            'last_betsize':36,
            'blind':37,
            'pot':38,
            'amount_to_call':39,
            'pot_odds':40,
            'player1_position':41,
            'player1_stacksize':42,
            'player1_street_total':43,
            'player1_status':44,
            'player2_position':45,
            'player2_stacksize':46,
            'player2_street_total':47,
            'player2_status':48,
            'ordinal': [0,10,30,31,32,34,37,40,41,44],
            'continuous': [1,23,25,26,28,29,31,32,35,36,38,39,42,43],
            'hand_board':[2,3,4,5,6,7,8,9] + [20,21,22,23,24,25,26,27,28,29],
            'villain_board':[12,13,14,15,16,17,18,19] + [20,21,22,23,24,25,26,27,28,29],
        }

        game_object = pdt.Globals.GameTypeDict[pdt.GameTypes.OMAHAHI]
        self.env_params = {
            'game':pdt.GameTypes.OMAHAHI,
            'betsizes': game_object.rule_params['betsizes'],
            'bet_type': game_object.rule_params['bettype'],
            'n_players': 2,
            'pot':1,
            'stacksize': game_object.state_params['stacksize'],
            'cards_per_player': game_object.state_params['cards_per_player'],
            'starting_street': game_object.starting_street,
            'global_mapping':global_mapping,
            'state_mapping':state_mapping,
            'obs_mapping':obs_mapping,
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
        env = Poker(self.env_params)

        state,obs,done,mask,betsize_mask = env.reset()
        assert state.ndim == 3
        assert obs.ndim == 3
        assert state.shape == (1,1,STATE_SHAPE)
        assert obs.shape == (1,1,OBS_SHAPE)
        assert state[0,0,env.state_mapping['street']] == self.env_params['starting_street']
        assert state[0,-1,env.state_mapping['hero_position']] == 1
        assert state[0,-1,env.state_mapping['hero_stacksize']] == self.env_params['stacksize']
        assert state[0,-1,env.state_mapping['player1_position']] == 1
        assert state[0,-1,env.state_mapping['player1_stacksize']] == self.env_params['stacksize']
        assert state[0,-1,env.state_mapping['player1_street_total']] == 0
        assert state[0,-1,env.state_mapping['player2_position']] == 0
        assert state[0,-1,env.state_mapping['player2_stacksize']] == self.env_params['stacksize']
        assert state[0,-1,env.state_mapping['player2_street_total']] == 0
        assert state[0,-1,env.state_mapping['last_action']] == 5
        assert state[0,-1,env.state_mapping['last_betsize']] == 0
        assert state[0,-1,env.state_mapping['last_position']] == 2
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
        assert state[:,1][:,env.state_mapping['hero_position']] == 0
        assert state[:,1][:,env.state_mapping['player2_position']] == 1
        assert state[:,1][:,env.state_mapping['last_position']] == 1
        assert state[:,1][:,env.state_mapping['last_action']] == 0
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
        assert state[:,-1][:,env.state_mapping['hero_position']] == 1
        assert state[:,-1][:,env.state_mapping['last_position']] == 0
        assert state[:,-1][:,env.state_mapping['last_action']] == 3
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
        env = Poker(self.env_params)
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
        assert state[:,-1][:,env.state_mapping['street']] == self.env_params['starting_street']
        assert state[:,-1][:,env.state_mapping['hero_position']] == 0
        assert state[:,-1][:,env.state_mapping['last_position']] == 1
        assert state[:,-1][:,env.state_mapping['last_action']] == 3
        assert state[:,-1][:,env.state_mapping['last_betsize']] == 1
        assert state[:,-1][:,env.state_mapping['hero_stacksize']] == self.env_params['stacksize']
        assert state[:,-1][:,env.state_mapping['player2_stacksize']] == self.env_params['stacksize'] - 1
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
        assert state[:,-1][:,env.state_mapping['hero_position']] == 1
        assert state[:,-1][:,env.state_mapping['last_position']] == 0
        assert state[:,-1][:,env.state_mapping['last_action']] == 4
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
        assert state[0,-1,env.state_mapping['blind']] == 0
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
        params = self.env_params
        params['starting_street'] = 0
        params['pot'] = 0
        env = Poker(self.env_params)
        state,obs,done,mask,betsize_mask = env.reset()
        assert env.players['SB'].stack == 4.5
        assert env.players['BB'].stack == 4.
        assert env.players['SB'].street_total == 0.5
        assert env.players['BB'].street_total == 1.
        assert state[0,-1,env.state_mapping['blind']] == 1
        assert state[:,-1][:,env.state_mapping['hero_position']] == 0
        assert state[:,-1][:,env.state_mapping['last_position']] == 1
        assert done == False

    def testStreetIncrement(self):
        params = self.env_params
        params['starting_street'] = 2
        params['pot'] = 1
        env = Poker(self.env_params)
        state,obs,done,mask,betsize_mask = env.reset()
        assert env.board[-2] == 0
        assert env.board[-1] == 0
        state,obs,done,mask,betsize_mask = env.step(ACTION_BET)
        state,obs,done,mask,betsize_mask = env.step(ACTION_CALL)
        assert env.street == 3
        assert env.board[-2] != 0
        state,obs,done,mask,betsize_mask = env.step(ACTION_BET)
        state,obs,done,mask,betsize_mask = env.step(ACTION_CALL)
        assert done == True
        del env
        params['starting_street'] = 0
        params['pot'] = 0
        env = Poker(self.env_params)
        state,obs,done,mask,betsize_mask = env.reset()
        state,obs,done,mask,betsize_mask = env.step(ACTION_CALL)
        assert state[:,-1][:,env.state_mapping['hero_position']] == 1
        assert state[:,-1][:,env.state_mapping['last_position']] == 0
        assert env.pot == 2
        state,obs,done,mask,betsize_mask = env.step(ACTION_RAISE)
        assert env.players['BB'].stack == 2
        assert env.players['SB'].stack == 4
        assert env.pot == 4
        assert state[:,-1][:,env.state_mapping['hero_position']] == 0
        assert state[:,-1][:,env.state_mapping['last_position']] == 1
        state,obs,done,mask,betsize_mask = env.step(ACTION_CALL)
        assert state[:,-1][:,env.state_mapping['hero_position']] == 1
        assert state[:,-1][:,env.state_mapping['player2_position']] == 0
        assert state[:,-1][:,env.state_mapping['last_position']] == 2
        assert state[:,-1][:,env.state_mapping['last_aggressive_position']] == 2
        assert env.street == 1
        state,obs,done,mask,betsize_mask = env.step(ACTION_CHECK)
        state,obs,done,mask,betsize_mask = env.step(ACTION_CHECK)
        assert env.street == 2
        state,obs,done,mask,betsize_mask = env.step(ACTION_CHECK)
        state,obs,done,mask,betsize_mask = env.step(ACTION_CHECK)
        assert env.street == 3
        state,obs,done,mask,betsize_mask = env.step(ACTION_CHECK)
        state,obs,done,mask,betsize_mask = env.step(ACTION_CHECK)
        assert done == True

    def testThreePlayers(self):
        params = copy.deepcopy(self.env_params)
        params['n_players'] = 3
        params['starting_street'] = 0
        params['pot'] = 0
        env = Poker(params)
        state,obs,done,mask,betsize_mask = env.reset()
        assert state[:,-1][:,env.state_mapping['hero_position']] == 2
        assert state[:,-1][:,env.state_mapping['player1_position']] == 2
        assert state[:,-1][:,env.state_mapping['player2_position']] == 0
        assert state[:,-1][:,env.state_mapping['player3_position']] == 1
        assert env.street == 0
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
        assert env.street == 1
        assert env.players.num_active_players == 2
        assert state[:,-1][:,env.state_mapping['hero_position']] == 1
        state,obs,done,mask,betsize_mask = env.step(ACTION_CHECK)
        state,obs,done,mask,betsize_mask = env.step(ACTION_CHECK)
        assert env.street == 2
        state,obs,done,mask,betsize_mask = env.step(ACTION_CHECK)
        state,obs,done,mask,betsize_mask = env.step(ACTION_CHECK)
        assert env.street == 3
        state,obs,done,mask,betsize_mask = env.step(ACTION_CHECK)
        state,obs,done,mask,betsize_mask = env.step(ACTION_CHECK)
        assert done == True
        assert env.players['SB'].stack == 4.5
        assert env.players['BB'].stack == 9
        assert env.players['BTN'].stack == 1.5
        del env
        params['n_players'] = 3
        params['starting_street'] = 0
        params['pot'] = 0
        env = Poker(params)
        state,obs,done,mask,betsize_mask = env.reset()
        assert state[:,-1][:,env.state_mapping['hero_position']] == 2
        state,obs,done,mask,betsize_mask = env.step(ACTION_CALL)
        assert state[:,-1][:,env.state_mapping['hero_position']] == 0
        assert env.players['SB'].street_total == 0.5
        state,obs,done,mask,betsize_mask = env.step(ACTION_CALL)
        assert env.players['SB'].street_total == 1
        assert state[:,-1][:,env.state_mapping['hero_position']] == 1
        state,obs,done,mask,betsize_mask = env.step(ACTION_CHECK)
        assert env.street == 1
        assert env.pot == 3
        state,obs,done,mask,betsize_mask = env.step(ACTION_CHECK)
        state,obs,done,mask,betsize_mask = env.step(ACTION_CHECK)
        state,obs,done,mask,betsize_mask = env.step(ACTION_CHECK)
        assert env.street == 2
        state,obs,done,mask,betsize_mask = env.step(ACTION_CHECK)
        state,obs,done,mask,betsize_mask = env.step(ACTION_CHECK)
        state,obs,done,mask,betsize_mask = env.step(ACTION_CHECK)
        assert env.street == 3
        state,obs,done,mask,betsize_mask = env.step(ACTION_CHECK)
        state,obs,done,mask,betsize_mask = env.step(ACTION_CHECK)
        state,obs,done,mask,betsize_mask = env.step(ACTION_CHECK)
        assert done == True
        assert env.players['SB'].stack == 7
        assert env.players['BB'].stack == 4
        assert env.players['BTN'].stack == 4.

    def testBetLimits(self):
        params = copy.deepcopy(self.env_params)
        # Limit
        params['bet_type'] = pdt.LimitTypes.LIMIT
        params['n_players'] = 3
        params['starting_street'] = 0
        params['pot'] = 0
        env = Poker(params)
        state,obs,done,mask,betsize_mask = env.reset()
        assert state[:,-1][:,env.state_mapping['pot']] == 1.5
        assert state[:,-1][:,env.state_mapping['hero_position']] == 2
        state,obs,done,mask,betsize_mask = env.step(ACTION_RAISE)
        assert state[:,-1][:,env.state_mapping['pot']] == 3.5
        assert env.players['BTN'].stack == 3
        assert env.players['BB'].stack == 4
        assert env.players['SB'].stack == 4.5
        assert env.players['SB'].street_total == 0.5
        assert state[:,-1][:,env.state_mapping['hero_position']] == 0
        assert state[:,-1][:,env.state_mapping['last_aggressive_betsize']] == 2
        assert env.street == 0
        state,obs,done,mask,betsize_mask = env.step(ACTION_RAISE)
        assert env.players['BTN'].stack == 3
        assert env.players['BB'].stack == 4
        assert env.players['SB'].stack == 2
        assert env.players['SB'].street_total == 3.
        assert state[:,-1][:,env.state_mapping['hero_position']] == 1
        assert state[:,-1][:,env.state_mapping['last_aggressive_betsize']] == 2.5
        assert state[:,-1][:,env.state_mapping['pot']] == 6
        assert env.street == 0
        state,obs,done,mask,betsize_mask = env.step(ACTION_CALL)
        assert state[:,-1][:,env.state_mapping['pot']] == 8
        assert state[:,-1][:,env.state_mapping['hero_position']] == 2
        assert env.street == 0
        state,obs,done,mask,betsize_mask = env.step(ACTION_CALL)
        assert state[:,-1][:,env.state_mapping['pot']] == 9
        assert state[:,-1][:,env.state_mapping['hero_position']] == 0
        assert env.street == 1
        del env
        params['bet_type'] = pdt.LimitTypes.POT_LIMIT
        params['n_players'] = 3
        params['starting_street'] = 0
        params['pot'] = 0
        params['stacksize'] = 100
        env = Poker(params)
        state,obs,done,mask,betsize_mask = env.reset()
        assert state[:,-1][:,env.state_mapping['hero_position']] == 2
        state,obs,done,mask,betsize_mask = env.step(ACTION_RAISE)
        assert state[:,-1][:,env.state_mapping['hero_position']] == 0
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
        params['starting_street'] = 0
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
        assert env.street == 1
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
        # params['starting_street'] = 0
        # params['pot'] = 0
        # env = Poker(params)
        # state,obs,done,mask,betsize_mask = env.reset()
        # del env
        # Pot Limit
        # Test reraise, preflop raise, sb raise preflop. raise vs bet

    def testAllin(self):
        params = copy.deepcopy(self.env_params)
        params['n_players'] = 3
        params['starting_street'] = 0
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
        assert env.players['BTN'].stack == 0
        assert env.players['BTN'].street_total == 0
        assert env.street == 3
        assert done == True

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
    return suite

if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(envTestSuite())