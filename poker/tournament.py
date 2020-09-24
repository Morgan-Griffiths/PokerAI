import os
import copy
import torch
import sys
import numpy as np

import models.network_config as ng
from models.networks import OmahaActor,CombinedNet
from models.network_config import NetworkConfig
import poker_env.datatypes as pdt
from poker_env.config import Config
from poker_env.env import Poker

class BetAgent(object):
    def __init__(self):
        pass

    def name(self):
        return 'baseline_evaluation'

    def __call__(self,state,action_mask,betsize_mask):
        if betsize_mask.sum() > 0:
            action = np.argmax(betsize_mask,axis=-1) + 3
        else:
            action = np.argmax(action_mask,axis=-1)
        actor_outputs = {
            'action':action,
            'action_category':int(np.where(action_mask > 0)[-1][-1]),
            'action_probs':torch.zeros(5).fill_(2.),
            'action_prob':torch.tensor([1.]),
            'betsize' : int(np.argmax(betsize_mask,axis=-1))
        }
        return actor_outputs

def tournament(env,agent1,agent2,model_names,training_params):
    agent_performance = {
        model_names[0]: {'SB':0,'BB':0},
        model_names[1]: {'SB':0,'BB':0}
    }
    for e in range(training_params['epochs']):
        sys.stdout.write('\r')
        if e % 2 == 0:
            agent_positions = {'SB':agent1,'BB':agent2}
            agent_loc = {'SB':model_names[0],'BB':model_names[1]}
        else:
            agent_positions = {'SB':agent2,'BB':agent1}
            agent_loc = {'SB':model_names[1],'BB':model_names[0]}
        state,obs,done,action_mask,betsize_mask = env.reset()
        while not done:
            actor_outputs = agent_positions[env.current_player](state,action_mask,betsize_mask)
            state,obs,done,action_mask,betsize_mask = env.step(actor_outputs)

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
    parser.add_argument('--game','-g',
                        default=pdt.GameTypes.OMAHAHI,
                        metavar=f"[{pdt.GameTypes.OMAHAHI},{pdt.GameTypes.HOLDEM}]",
                        help='Picks which type of poker env to play')
    parser.add_argument('--network','-n',
                        dest='network_type',
                        default='combined',
                        metavar=f"['combined','dual]",
                        help='Selects network type')
    parser.add_argument('--model','-m',
                        dest='network',
                        default='CombinedNet',
                        metavar=f"['CombinedNet','OmahaActor']",
                        help='Selects model type')

    args = parser.parse_args()

    print(f'Args {args}')
    config = Config()
    game_object = pdt.Globals.GameTypeDict[pdt.GameTypes.OMAHAHI]

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
    print(f'Environment Parameters: Starting street: {env_params["starting_street"]},\
        Stacksize: {env_params["stacksize"]},\
        Pot: {env_params["pot"]},\
        Bettype: {env_params["bet_type"]},\
        Betsizes: {env_params["betsizes"]}')
    env = Poker(env_params)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_params = config.training_params
    training_params['epochs'] = 500
    network_params = {
        'game':pdt.GameTypes.OMAHAHI,
        'maxlen':config.maxlen,
        'state_mapping':config.state_mapping,
        'obs_mapping':config.obs_mapping,
        'embedding_size':128,
        'transformer_in':1280,
        'transformer_out':128,
        'device':device,
    }

    nS = env.state_space
    nO = env.observation_space
    nA = env.action_space
    nB = env.betsize_space
    model_name = 'RL_actor' if args.network_type == 'dual' else 'RL_combined'
    print(f'Environment: State Space {nS}, Obs Space {nO}, Action Space {nA}, Betsize Space {nB}')
    print(f'Evaluating {model_name}')
    seed = 154

    if args.network_type == 'dual':
        trained_model = OmahaActor(seed,nS,nA,nB,network_params).to(device)
    else:
        trained_model = CombinedNet(seed,nS,nA,nB,network_params).to(device)
    trained_model.load_state_dict(torch.load(os.path.join(training_params['save_dir'],model_name)))
    baseline_evaluation = BetAgent()

    model_names = ['baseline_evaluation','trained_model']

    results = tournament(env,baseline_evaluation,trained_model,model_names,training_params)
    print(results)
    print(f"{model_names[0]}: {results[model_names[0]]['SB'] + results[model_names[0]]['BB']}")
    print(f"{model_names[1]}: {results[model_names[1]]['SB'] + results[model_names[1]]['BB']}")