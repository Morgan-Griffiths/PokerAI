import torch
import unittest
import numpy as np
import os

from kuhn.config import Config
from kuhn.env import Poker
from kuhn.data_classes import Card,Evaluator
import kuhn.datatypes as pdt
import poker.datatypes as hdt
from utils.cardlib import winner,holdem_winner,encode

def run_kuhn_env(env,case):
    step = 0
    last_state,state,obs,done,mask,betsize_mask = env.reset()
    while not done:
        action,action_prob,action_probs = case[step]
        actor_output = {
            'action':action,
            'action_prob':action_prob,
            'action_probs':action_probs
        }
        last_state,state,obs,done,mask,betsize_mask = env.step(actor_output)
        step += 1
    return env

class TestEnv(unittest.TestCase):
    @classmethod
    def setUp(self):
        gametype = 'kuhn'

        self.config = Config()
        self.params = {'game':gametype}
        self.params['state_params'] = pdt.Globals.GameTypeDict[gametype].state_params
        self.params['rule_params'] = pdt.Globals.GameTypeDict[gametype].rule_params
        logs = torch.randn(4)
        complete_logs = torch.randn(4)
        actions = ['check','bet','call','fold','raise']
        self.actions = [[(torch.tensor([pdt.Globals.REVERSE_ACTION_ORDER[act]]),logs,complete_logs) for act in actions]]
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
            env = run_kuhn_env(env,case)
        
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
                action,action_logprobs,complete_probs = case[step]
                state,obs,done = env.step(action,action_logprobs,complete_probs)
                step += 1
                if step == 1:
                    scenarios[i].append(env.save_scenario())
            for scenario in scenarios[i]:
                state,obs,done = env.reset()
                env.load_scenario(scenario)
                env = run_kuhn_env(env,case)
                
        for i in range(len(self.actions)):
            print(f'Scenario {i}')
            env.load_scenario(scenarios[i][0])
            print('env.pot.value',env.pot.value,self.scenario_verification[i]['pot'])
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
            env = run_kuhn_env(env,case)
            ml_inputs = env.ml_inputs()
            # print(f'ml_inputs {ml_inputs}')


    def testEnvFunctions(self):
        env = Poker(self.params)
        ###  Testing Betsizes  ###
        actions = [0,1,2,3,4]
        actions = [torch.tensor([act]) for act in actions]
        betsizes = [0,0,0,0,0]
        betsizes = [torch.tensor([act]) for act in betsizes]
        answers = [[0,0,0],[0,0,0],[0,0,1],[torch.tensor([2.]),torch.tensor([2.]),torch.tensor([1.])],[torch.tensor([2.]),torch.tensor([2.]),torch.tensor([2.])]]
        i = 0
        for act,bet in zip(actions,betsizes):
            self.assertEqual(env.return_potlimit_betsize(act,bet),answers[i][0])
            self.assertEqual(env.return_nolimit_betsize(act,bet),answers[i][1])
            self.assertEqual(env.return_limit_betsize(act,bet),answers[i][2])
            i += 1
        ###  Testing Betsizes  ###
        states = [[3,5,0],[2,3,1],[1,4,2],[2,0,0]]
        states = [torch.tensor([state]) for state in states]
        answers = [1.,0.,0.,1.]
        for i,state in enumerate(states):
            self.assertEqual(env.return_betsizes(state),answers[i])
        action_answers = [torch.tensor([1., 0., 0., 1.]),torch.tensor([0., 1., 1., 0.]),torch.tensor([0., 1., 1., 0.]),torch.tensor([1., 0., 0., 1.])]
        betsize_answers = [torch.tensor([1.]),torch.tensor([0.]),torch.tensor([0.]),torch.tensor([1.])]
        for i,state in enumerate(states):
            available_categories,available_betsizes = env.action_mask(state)
            print(env.action_mask(state))
            self.assertTrue(np.array_equal(available_categories.numpy(),action_answers[i].numpy()))
            self.assertTrue(np.array_equal(available_betsizes.numpy(),betsize_answers[i].numpy()))
        # for i,case in enumerate(self.actions):
        #     print(f'Case {i}')
        #     env = run_kuhn_env(env,case)
        #     ml_inputs = env.ml_inputs()
            # print(f'ml_inputs {ml_inputs}')

    def testEvaluations(self):
        omaha = Evaluator(hdt.GameTypes.OMAHAHI)
        holdem = Evaluator(hdt.GameTypes.HOLDEM)
        kuhn = Evaluator(pdt.GameTypes.KUHN)

        Q = Card(1,None)
        K = Card(2,None)
        assert(kuhn([[K],[Q]]) == 0)
        assert(kuhn([[Q],[K]]) == 1)
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

    def TestRlEnvironments(self):
        os.system('python kuhn_main.py --env kuhn -e 2 --no-clean --no-store')
        os.system('python kuhn_main.py --env complexkuhn -e 2 --no-clean --no-store')
        os.system('python kuhn_main.py --env betsizekuhn -e 2 --no-clean --no-store')
        os.system('python kuhn_main.py --env historicalkuhn -e 2 --no-clean --no-store')
        os.system('python kuhn_main.py --env historicalkuhn -e 5 --parallel --no-clean --no-store')
        # os.system('python main.py --env holdem -e 10 --no-clean --no-store')
        # os.system('python main.py --env multistreetholdem -e 10 --no-clean --no-store')
        # os.system('python main.py --env omaha -e 10 --no-clean --no-store')

def kuhnTestSuite():
    suite = unittest.TestSuite()
    suite.addTest(TestEnv('testRun'))
    suite.addTest(TestEnv('testRepresentations'))
    suite.addTest(TestEnv('testEvaluations'))
    suite.addTest(TestEnv('testEnvFunctions'))
    # suite.addTest(TestEnv('TestRlEnvironments'))
    # suite.addTest(TestEnv('testScenario'))
    return suite

if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(kuhnTestSuite())