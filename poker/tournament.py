import os
import poker.datatypes as pdt
from poker.config import Config
import models.network_config as ng
from agents.agent import BetAgent,FullAgent
from models.network_config import NetworkConfig
from poker.multistreet_env import MSPoker
import copy
import torch
import sys

def tournament(env,agent1,agent2,training_params):
    agent_performance = {
        'agent1': {'SB':0,'BB':0},
        'agent2': {'SB':0,'BB':0}
    }
    for e in range(training_params['epochs']):
        sys.stdout.write('\r')
        if e % 2 == 0:
            agent_positions = {'SB':agent1,'BB':agent2}
            agent_loc = {'SB':'agent1','BB':'agent2'}
        else:
            agent_positions = {'SB':agent2,'BB':agent1}
            agent_loc = {'SB':'agent2','BB':'agent1'}
        state,obs,done,mask,betsize_mask = env.reset()
        while not done:
            actor_outputs = agent_positions[env.current_player](state,mask,betsize_mask)
            state,obs,done,mask,betsize_mask = env.step(actor_outputs)
        ml_inputs = env.ml_inputs()
        agent_performance[agent_loc['SB']]['SB'] += ml_inputs['SB']['rewards'][-1][-1]
        agent_performance[agent_loc['BB']]['BB'] += ml_inputs['BB']['rewards'][-1][-1]
        sys.stdout.write("[%-60s] %d%%" % ('='*(60*(e+1)//training_params['epochs']), (100*(e+1)//training_params['epochs'])))
        sys.stdout.flush()
        sys.stdout.write(", epoch %d"% (e+1))
        sys.stdout.flush()
    return agent_performance

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description=
        """
        Run tournaments between bots.\n
        """)
    parser.add_argument('--game',
                        default=pdt.GameTypes.OMAHAHI,
                        metavar=f"[{pdt.GameTypes.OMAHAHI},{pdt.GameTypes.HOLDEM}]",
                        help='Picks which type of poker env to play')

    args = parser.parse_args()
    config = Config()
    game_object = pdt.Globals.GameTypeDict[args.game]
    
    params = {'game':args.game}
    params['state_params'] = game_object.state_params
    params['maxlen'] = config.maxlen
    params['rule_params'] = game_object.rule_params
    training_params = config.training_params
    agent_params = config.agent_params
    env_networks = NetworkConfig.EnvModels[args.game]
    agent_params['network'] = env_networks['actor']
    agent_params['actor_network'] = env_networks['actor']
    agent_params['critic_network'] = env_networks['critic']['q']
    agent_params['mapping'] = params['rule_params']['mapping']
    agent_params['max_reward'] = params['state_params']['stacksize'] + params['state_params']['pot']
    agent_params['min_reward'] = params['state_params']['stacksize']
    agent_params['epochs'] = 0
    params['rule_params']['network_output'] = 'flat'
    params['rule_params']['betsizes'] = pdt.Globals.BETSIZE_DICT[2]
    agent_params['network_output'] = 'flat'
    agent_params['embedding_size'] = 32
    agent_params['maxlen'] = config.maxlen
    agent_params['game'] = args.game
    params['starting_street'] = game_object.starting_street
    training_params['epochs'] = 500

    env = MSPoker(params)

    nS = env.state_space
    nO = env.observation_space
    nA = env.action_space
    nB = env.rules.num_betsizes
    print(f'Environment: State Space {nS}, Obs Space {nO}, Action Space {nA}')
    seed = 154

    agent1 = FullAgent(nS,nO,nA,nB,seed,agent_params)
    agent1.load_weights(os.path.join(training_params['save_dir'],'Holdem'))

    agent2 = BetAgent()

    results = tournament(env,agent1,agent2,training_params)
    print(results)
    print(f"Agent1: {results['agent1']['SB'] + results['agent1']['BB']}")
    print(f"Agent2: {results['agent2']['SB'] + results['agent2']['BB']}")