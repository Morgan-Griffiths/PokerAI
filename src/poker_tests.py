import torch
import unittest
import numpy as np
import os

from poker.config import Config
from poker.multistreet_env import MSPoker
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
    'betsize':torch.Tensor([1])
}
ACTION_RAISE = {
    'action':torch.Tensor([4]).long(),
    'action_category':torch.Tensor([4]).long(),
    'action_probs':torch.zeros(5).fill_(0.2),
    'action_prob':torch.Tensor([0.2]),
    'betsize':torch.Tensor([1])
}

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
        self.gametype = 'holdem'
        self.game_object = pdt.Globals.GameTypeDict[self.gametype]
        self.config = Config()
        env_params = {'game':self.gametype}
        env_params['state_params'] = self.game_object.state_params
        env_params['rule_params'] = self.game_object.rule_params
        env_params['rule_params']['network_output'] = 'flat'
        env_params['rule_params']['betsizes'] = pdt.Globals.BETSIZE_DICT[2]
        env_params['starting_street'] = 3
        env_params['maxlen'] = 10
        env_params['state_params']['pot'] = 1.
        env_params['state_params']['stacksize'] = 2
        self.env_params = env_params

    def testInitialization(self):
        env = MSPoker(self.env_params)
        assert env.street == self.env_params['starting_street']
        assert env.game == self.env_params['game']
        assert len(env.stacksizes) == self.env_params['state_params']['n_players']
        assert env.stacksizes[0] == self.env_params['state_params']['stacksize']
        assert env.pot.value == self.env_params['state_params']['pot']
        assert len(env.players.players['SB'].hand) == self.env_params['state_params']['cards_per_player']
        assert len(env.players.players['BB'].hand) == self.env_params['state_params']['cards_per_player']
        assert len(env.board) == pdt.Globals.INITIALIZE_BOARD_CARDS[self.env_params['starting_street']]
        assert len(env.deck) == 52 - (self.env_params['state_params']['cards_per_player'] * self.env_params['state_params']['n_players'] + pdt.Globals.INITIALIZE_BOARD_CARDS[self.env_params['starting_street']]) 

    def testReset(self):
        env = MSPoker(self.env_params)

        state,obs,done,mask,betsize_mask = env.reset()
        assert state.ndim == 3
        assert obs.ndim == 3
        assert state.size() == (1,10,23)
        assert obs.size() == (1,10,27)
        assert state[:,0][:,env.db_mapping['state']['street']] == self.env_params['starting_street']
        assert state[:,0][:,env.db_mapping['state']['hero_position']] == 0
        assert state[:,0][:,env.db_mapping['state']['vil_position']] == 1
        assert state[:,0][:,env.db_mapping['state']['previous_action']] == 5
        assert state[:,0][:,env.db_mapping['state']['hero_stack']] == self.env_params['state_params']['stacksize']
        assert state[:,0][:,env.db_mapping['state']['villain_stack']] == self.env_params['state_params']['stacksize']
        assert state[:,0][:,env.db_mapping['state']['amnt_to_call']] == 0
        assert state[:,0][:,env.db_mapping['state']['pot_odds']] == 0
        assert state[:,0].sum() > 0
        assert state[:,1:].sum() == 0
        assert done == False
        assert np.array_equal(mask.numpy(),np.array([1., 0., 0., 1., 0.]))
        assert np.array_equal(betsize_mask.numpy(),np.array([1.,1.]))

    def testCheckCheck(self):
        env = MSPoker(self.env_params)
        state,obs,done,mask,betsize_mask = env.reset()
        state,obs,done,mask,betsize_mask = env.step(ACTION_CHECK)
        assert state.ndim == 3
        assert obs.ndim == 3
        assert state.size() == (1,10,23)
        assert obs.size() == (1,10,27)
        assert state[:,0].sum() > 0
        assert state[:,1].sum() > 0
        assert state[:,1][:,env.db_mapping['state']['street']] == self.env_params['starting_street']
        assert state[:,1][:,env.db_mapping['state']['hero_position']] == 1
        assert state[:,1][:,env.db_mapping['state']['vil_position']] == 0
        assert state[:,1][:,env.db_mapping['state']['previous_action']] == 0
        assert state[:,1][:,env.db_mapping['state']['hero_stack']] == self.env_params['state_params']['stacksize']
        assert state[:,1][:,env.db_mapping['state']['villain_stack']] == self.env_params['state_params']['stacksize']
        assert state[:,1][:,env.db_mapping['state']['amnt_to_call']] == 0
        assert state[:,1][:,env.db_mapping['state']['pot_odds']] == 0
        assert state[:,2:].sum() == 0
        assert done == False
        assert np.array_equal(mask.numpy(),np.array([1., 0., 0., 1., 0.]))
        assert np.array_equal(betsize_mask.numpy(),np.array([1.,1.]))

        state,obs,done,mask,betsize_mask = env.step(ACTION_CHECK)
        assert done == True

    def testCheckBetFold(self):
        env = MSPoker(self.env_params)
        state,obs,done,mask,betsize_mask = env.reset()
        state,obs,done,mask,betsize_mask = env.step(ACTION_CHECK)
        assert state.ndim == 3
        assert obs.ndim == 3
        assert state.size() == (1,10,23)
        assert obs.size() == (1,10,27)
        assert state[:,0].sum() > 0
        assert state[:,1].sum() > 0
        assert state[:,1][:,env.db_mapping['state']['street']] == self.env_params['starting_street']
        assert state[:,1][:,env.db_mapping['state']['hero_position']] == 1
        assert state[:,1][:,env.db_mapping['state']['vil_position']] == 0
        assert state[:,1][:,env.db_mapping['state']['previous_action']] == 0
        assert state[:,1][:,env.db_mapping['state']['hero_stack']] == self.env_params['state_params']['stacksize']
        assert state[:,1][:,env.db_mapping['state']['villain_stack']] == self.env_params['state_params']['stacksize']
        assert state[:,1][:,env.db_mapping['state']['amnt_to_call']] == 0
        assert state[:,1][:,env.db_mapping['state']['pot_odds']] == 0
        assert state[:,2:].sum() == 0
        assert done == False
        assert np.array_equal(mask.numpy(),np.array([1., 0., 0., 1., 0.]))
        assert np.array_equal(betsize_mask.numpy(),np.array([1.,1.]))

        state,obs,done,mask,betsize_mask = env.step(ACTION_BET)
        assert state.ndim == 3
        assert obs.ndim == 3
        assert state.size() == (1,10,23)
        assert obs.size() == (1,10,27)
        assert state[:,0].sum() > 0
        assert state[:,1].sum() > 0
        assert state[:,2].sum() > 0
        assert state[:,2][:,env.db_mapping['state']['street']] == self.env_params['starting_street']
        assert state[:,2][:,env.db_mapping['state']['hero_position']] == 0
        assert state[:,2][:,env.db_mapping['state']['vil_position']] == 1
        assert state[:,2][:,env.db_mapping['state']['previous_action']] == 3
        assert state[:,2][:,env.db_mapping['state']['hero_stack']] == self.env_params['state_params']['stacksize']
        assert state[:,2][:,env.db_mapping['state']['villain_stack']] == self.env_params['state_params']['stacksize'] - 1
        assert state[:,2][:,env.db_mapping['state']['amnt_to_call']] == 1
        self.assertAlmostEqual(state[:,2][:,env.db_mapping['state']['pot_odds']].numpy()[0][0],0.333,places=2)
        assert state[:,3:].sum() == 0
        assert done == False
        assert np.array_equal(mask.numpy(),np.array([0., 1., 1., 0., 1.]))
        assert np.array_equal(betsize_mask.numpy(),np.array([1.,0.]))

        state,obs,done,mask,betsize_mask = env.step(ACTION_FOLD)
        assert done == True

    def testBetRaiseCall(self):
        env = MSPoker(self.env_params)
        state,obs,done,mask,betsize_mask = env.reset()
        state,obs,done,mask,betsize_mask = env.step(ACTION_BET)
        assert state.ndim == 3
        assert obs.ndim == 3
        assert state.size() == (1,10,23)
        assert obs.size() == (1,10,27)
        assert state[:,0].sum() > 0
        assert state[:,1].sum() > 0
        assert state[:,1][:,env.db_mapping['state']['street']] == self.env_params['starting_street']
        assert state[:,1][:,env.db_mapping['state']['hero_position']] == 1
        assert state[:,1][:,env.db_mapping['state']['vil_position']] == 0
        assert state[:,1][:,env.db_mapping['state']['previous_action']] == 3
        assert state[:,1][:,env.db_mapping['state']['hero_stack']] == self.env_params['state_params']['stacksize']
        assert state[:,1][:,env.db_mapping['state']['villain_stack']] == self.env_params['state_params']['stacksize'] - 1
        assert state[:,1][:,env.db_mapping['state']['amnt_to_call']] == 1
        self.assertAlmostEqual(state[:,1][:,env.db_mapping['state']['pot_odds']].numpy()[0][0],0.333,places=2)
        assert state[:,2:].sum() == 0
        assert done == False
        assert np.array_equal(mask.numpy(),np.array([0., 1., 1., 0., 1.]))
        assert np.array_equal(betsize_mask.numpy(),np.array([1.,0.]))

        state,obs,done,mask,betsize_mask = env.step(ACTION_RAISE)
        assert state.ndim == 3
        assert obs.ndim == 3
        assert state.size() == (1,10,23)
        assert obs.size() == (1,10,27)
        assert state[:,0].sum() > 0
        assert state[:,1].sum() > 0
        assert state[:,2].sum() > 0
        assert state[:,2][:,env.db_mapping['state']['street']] == self.env_params['starting_street']
        assert state[:,2][:,env.db_mapping['state']['hero_position']] == 0
        assert state[:,2][:,env.db_mapping['state']['vil_position']] == 1
        assert state[:,2][:,env.db_mapping['state']['previous_action']] == 4
        assert state[:,2][:,env.db_mapping['state']['hero_stack']] == self.env_params['state_params']['stacksize'] - 1
        assert state[:,2][:,env.db_mapping['state']['villain_stack']] == self.env_params['state_params']['stacksize'] - 2
        assert state[:,2][:,env.db_mapping['state']['amnt_to_call']] == 1
        self.assertAlmostEqual(state[:,2][:,env.db_mapping['state']['pot_odds']].numpy()[0][0],0.2,places=2)
        assert state[:,3:].sum() == 0
        assert done == False
        assert np.array_equal(mask.numpy(),np.array([0., 1., 1., 0., 0.]))
        assert np.array_equal(betsize_mask.numpy(),np.array([0.,0.]))

        state,obs,done,mask,betsize_mask = env.step(ACTION_CALL)
        assert done == True

    def testML(self):
        env = MSPoker(self.env_params)
        state,obs,done,mask,betsize_mask = env.reset()
        state,obs,done,mask,betsize_mask = env.step(ACTION_BET)
        state,obs,done,mask,betsize_mask = env.step(ACTION_RAISE)
        state,obs,done,mask,betsize_mask = env.step(ACTION_CALL)
        ml_inputs = env.ml_inputs()
        # print('ml_inputs',ml_inputs['SB'])
        # asserts

    def testRun(self):
        env = MSPoker(self.env_params)

    def testEvaluations(self):
        holdem = Evaluator(pdt.GameTypes.HOLDEM)
        omaha = Evaluator(pdt.GameTypes.OMAHAHI)

        hand = [Card(14,'c'),Card(2,'c'),Card(2,'d'),Card(5,'c')]
        hand2 = [Card(7,'s'),Card(5,'c'),Card(14,'h'),Card(10,'h')]
        PLO_board = [Card(10,'c'),Card(2,'h'),Card(4,'c'),Card(13,'c'),Card(4,'h')]
        assert(omaha([hand,hand2,PLO_board]) == 1)
        assert(omaha([hand2,hand,PLO_board]) == -1)
        # Holdem
        holdem_hand = [Card(14,'c'),Card(2,'s')]
        holdem_hand2 = [Card(12,'c'),Card(5,'s')]
        holdem_board = [Card(10,'c'),Card(2,'c'),Card(4,'c'),Card(13,'c'),Card(4,'h')]
        assert(holdem([holdem_hand,holdem_hand2,holdem_board]) == 1)
        assert(holdem([holdem_hand2,holdem_hand,holdem_board]) == -1)

def pokerTestSuite():
    suite = unittest.TestSuite()
    suite.addTest(TestEnv('testInitialization'))
    suite.addTest(TestEnv('testReset'))
    suite.addTest(TestEnv('testCheckCheck'))
    suite.addTest(TestEnv('testCheckBetFold'))
    suite.addTest(TestEnv('testBetRaiseCall'))
    suite.addTest(TestEnv('testEvaluations'))
    suite.addTest(TestEnv('testML'))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(pokerTestSuite())