from config import Config
from environment import Poker
from data_classes import Card
import torch
import unittest
import numpy as np


class TestEnv(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.config = Config()
        self.params = self.config.params
        logs = torch.randn(4)
        torch_action_dict = {'check':torch.tensor(0),'bet':torch.tensor(1),'call':torch.tensor(2),'fold':torch.tensor(3)}
        self.actions = [[(torch_action_dict['bet'],logs),(torch_action_dict['call'],logs)],
                [(torch_action_dict['check'],logs)],
                [(torch_action_dict['bet'],logs),(torch_action_dict['fold'],logs)]]
        self.scenario_verification = {
            0:{
                'pot':2,
                'stack':0,
                'game_turn':1,
            },
            1:{
                'pot':1,
                'stack':2,
                'game_turn':1,
            },
            2:{
                'pot':2,
                'stack':0,
                'game_turn':1,
            }
        }
        Q = Card(1,None)
        K = Card(2,None)
        A = Card(3,None)
        self.hands = [[K,Q],[K,Q],[K,Q]]

    def testRun(self):
        env = Poker(self.params)
        for i,case in enumerate(self.actions):
            print(f'Case {i}')
            step = 0
            state,obs,done = env.reset()
            while not done:
                action,action_logprobs = case[step]
                state,obs,done = env.step(action,action_logprobs)
                step += 1
        
    def testScenario(self):
        env = Poker(self.params)
        scenarios = {}
        for i,case in enumerate(self.actions):
            print(f'Case {i}')
            scenarios[i] = []
            step = 0
            state,obs,done = env.reset()
            env.players.update_hands(self.hands[i])
            while not done:
                action,action_logprobs = case[step]
                state,obs,done = env.step(action,action_logprobs)
                step += 1
                if step == 1:
                    scenarios[i].append(env.save_scenario())
            for scenario in scenarios[i]:
                state,obs,done = env.reset()
                env.load_scenario(scenario)
                step = 0
                while not done:
                    action,action_logprobs = case[step]
                    state,obs,done = env.step(action,action_logprobs)
                    step += 1
                
        for i in range(len(self.actions)):
            print(f'Scenario {i}')
            env.load_scenario(scenarios[i][0])
            print('env.pot.value',env.pot.value)
            assert(env.pot.value == self.scenario_verification[i]['pot'])
            player = env.players.get_player('SB')
            print('player.stack',player.stack,self.scenario_verification[i]['stack'])
            assert(player.stack == self.scenario_verification[i]['stack'])
            print('env.game_turn.value',env.game_turn.value,self.scenario_verification[i]['game_turn'])
            assert(env.game_turn.value == self.scenario_verification[i]['game_turn'])

    def testRepresentations(self):
        env = Poker(self.params)
        for i,case in enumerate(self.actions):
            print(f'Case {i}')
            step = 0
            state,obs,done = env.reset()
            while not done:
                action,action_logprobs = case[step]
                state,obs,done = env.step(action,action_logprobs)
                step += 1
            ml_inputs = env.ml_inputs()
            print(f'ml_inputs {ml_inputs}')


def envTestSuite():
    suite = unittest.TestSuite()
    suite.addTest(TestEnv('testRun'))
    suite.addTest(TestEnv('testScenario'))
    suite.addTest(TestEnv('testRepresentations'))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(envTestSuite())