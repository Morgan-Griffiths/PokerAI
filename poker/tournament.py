import os
import copy
import torch
import sys
import numpy as np
import time
from itertools import combinations
from prettytable import PrettyTable

import models.network_config as ng
from models.networks import OmahaActor,CombinedNet,BetAgent
from models.network_config import NetworkConfig
import poker_env.datatypes as pdt
from poker_env.config import Config
from poker_env.env import Poker
from utils.utils import load_paths

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
                        default='dual',
                        metavar=f"['combined','dual]",
                        help='Selects network type')
    parser.add_argument('--model','-m',
                        dest='network',
                        default='CombinedNet',
                        metavar=f"['CombinedNet','OmahaActor']",
                        help='Selects model type')
    parser.add_argument('--epochs','-e',
                        dest='epochs',
                        default=500,
                        type=int,
                        help='How many hands to evaluate on')
    parser.add_argument('--tourney','-t',
                        dest='tourney',
                        default='baseline',
                        type=str,
                        metavar=f'[roundrobin,latest,baseline]',
                        help='What kind of tournament to run')

    args = parser.parse_args()
    tic = time.time()

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
    training_params['epochs'] = args.epochs
    training_params['save_dir']:os.path.join(os.getcwd(),'checkpoints/training_run')
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
    model_name = 'OmahaActorFinal' if args.network_type == 'dual' else 'OmahaCombinedFinal'
    print(f'Environment: State Space {nS}, Obs Space {nO}, Action Space {nA}, Betsize Space {nB}')
    print(f'Evaluating {model_name}')
    seed = 154

    if args.network_type == 'dual':
        trained_model = OmahaActor(seed,nS,nA,nB,network_params).to(device)
    else:
        trained_model = CombinedNet(seed,nS,nA,nB,network_params).to(device)

    if args.tourney == 'latest':
        """Takes the latest network weights and evals vs all the previous ones or the last N"""
        # load all file paths
        weight_paths = load_paths(training_params['save_dir'])
        model_names = weight_paths.keys()
        latest_actor = model_names[-1]
        latest_net = OmahaActor(seed,nS,nA,nB,network_params).to(device)
        latest_net.load_state_dict(torch.load(weight_paths[latest_actor]))
        matchups = [(latest_actor,model) for model in model_names[:-1]]
        # create array to store results
        result_array = np.zeros(len(matchups))
        data_row_dict = {model:i for i,model in enumerate(model_names[:-1])}
        for match in matchups:
            net2 = OmahaActor(seed,nS,nA,nB,network_params).to(device)
            net2_path = weight_paths[match[1]]
            net2.load_state_dict(torch.load(net2_path))
            results = tournament(env,net1,net2,match,training_params)
            result_array[data_row_dict[match[0]]] = results[match[0]]['SB'] + results[match[0]]['BB']
        # Create Results Table
        table = PrettyTable(["Model Name", *model_names[:-1]])
        table.add_row([latest_actor,*result_array])
        print(table)

    elif args.tourney == 'roundrobin':
        """Runs all saved weights (in training_run folder) against each other in a round robin"""
        # load all file paths
        weight_paths = load_paths(training_params['save_dir'])
        # all combinations
        model_names = weight_paths.keys()
        matchups = list(combinations(model_names,2))
        # create array to store results
        result_array = np.zeros((len(matchups),len(matchups)))
        data_row_dict = {model:i for i,model in enumerate(model_names)}
        for match in matchups:
            net1 = OmahaActor(seed,nS,nA,nB,network_params).to(device)
            net2 = OmahaActor(seed,nS,nA,nB,network_params).to(device)
            net1_path = weight_paths[match[0]]
            net2_path = weight_paths[match[1]]
            net1.load_state_dict(torch.load(net1_path))
            net2.load_state_dict(torch.load(net2_path))
            results = tournament(env,net1,net2,match,training_params)
            result_array[data_row_dict[match[0]]] = results[match[0]]['SB'] + results[match[0]]['BB']
            result_array[data_row_dict[match[1]]] = results[match[1]]['SB'] + results[match[1]]['BB']
        # Create Results Table
        table = PrettyTable(["Model Name", *model_names])
        for i,model in enumerate(model_names):
            row = list(result_array[i])
            row[i] = 'x'
            table.add_row([model,*row])
        print(table)
    else:
        trained_model.load_state_dict(torch.load(os.path.join(training_params['save_dir'],model_name)))
        baseline_evaluation = BetAgent()
        model_names = ['baseline_evaluation','trained_model']
        results = tournament(env,baseline_evaluation,trained_model,model_names,training_params)
        print(results)

        print(f"{model_names[0]}: {results[model_names[0]]['SB'] + results[model_names[0]]['BB']}")
        print(f"{model_names[1]}: {results[model_names[1]]['SB'] + results[model_names[1]]['BB']}")
        toc = time.time()
        print(f'Tournament completed in {(toc-tic)/60} minutes')