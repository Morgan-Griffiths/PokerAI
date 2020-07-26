import os
import poker.datatypes as pdt
from poker.config import Config
import models.network_config as ng
from models.networks import OmahaActor
from agents.agent import BetAgent,FullAgent
from models.network_config import NetworkConfig
from poker.env import Poker
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
        rewards = env.player_rewards()
        agent_performance[agent_loc['SB']]['SB'] += rewards['SB']
        agent_performance[agent_loc['BB']]['BB'] += rewards['BB']
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
    
    env_params = {
        'game':pdt.GameTypes.OMAHAHI,
        'betsizes': game_object.rule_params['betsizes'],
        'bet_type': game_object.rule_params['bettype'],
        'n_players': 2,
        'pot':1,
        'stacksize': game_object.state_params['stacksize'],
        'cards_per_player': game_object.state_params['cards_per_player'],
        'starting_street': game_object.starting_street,
        'global_mapping':config.global_mapping,
        'state_mapping':config.state_mapping,
        'obs_mapping':config.obs_mapping,
        'shuffle':True
    }

    training_params = config.training_params
    agent_params = config.agent_params
    env_networks = NetworkConfig.EnvModels[args.game]
    agent_params['network'] = env_networks['actor']
    agent_params['actor_network'] = env_networks['actor']
    agent_params['critic_network'] = env_networks['critic']['q']
    agent_params['mapping'] = env_params['state_mapping']
    agent_params['max_reward'] = env_params['stacksize'] + env_params['pot']
    agent_params['min_reward'] = env_params['stacksize']
    agent_params['epochs'] = 0
    agent_params['network_output'] = 'flat'
    agent_params['embedding_size'] = 32
    agent_params['maxlen'] = config.maxlen
    agent_params['game'] = args.game
    training_params['epochs'] = 500

    env = Poker(env_params)

    nS = env.state_space
    nO = env.observation_space
    nA = env.action_space
    nB = env.betsize_space
    print(f'Environment: State Space {nS}, Obs Space {nO}, Action Space {nA}')
    seed = 154
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'

    network_params = {
        'game':pdt.GameTypes.OMAHAHI,
        'maxlen':config.maxlen,
        'state_mapping':config.state_mapping,
        'embedding_size':128,
        'device':device,
        'path': os.path.join(os.getcwd(),'checkpoints/RL/omaha_hi_actor')
    }

    actor = OmahaActor(seed,nS,nA,nB,network_params)
    actor.load_state_dict(torch.load(network_params['path']))
    actor.eval()

    agent2 = BetAgent()

    results = tournament(env,actor,agent2,training_params)
    print(results)
    print(f"Agent1: {results['agent1']['SB'] + results['agent1']['BB']}")
    print(f"Agent2: {results['agent2']['SB'] + results['agent2']['BB']}")