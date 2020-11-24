import os
import copy
import torch
import sys
import numpy as np
import time
from itertools import combinations
from collections import defaultdict,deque
from prettytable import PrettyTable
from random import shuffle

import models.network_config as ng
from models.networks import OmahaActor,CombinedNet,BetAgent
from models.network_config import NetworkConfig
from models.model_utils import hardcode_handstrength,load_weights
import poker_env.datatypes as pdt
from poker_env.config import Config

from poker_env.env import Poker
from utils.utils import load_paths,grep,return_latest_baseline_path,bin_by_handstrength

def ELO(A_rating,A_score,B_rating,B_score,K=32):
    A_expected_score = 1 / 1 + 10**((A-B)/400)
    B_expected_score = 1 / 1 + 10**((B-A)/400)
    A_new_rating = A_rating + K * (A_score-A_expected_score)
    B_new_rating = B_rating + K * (B_score-B_expected_score)
    return A_new_rating,B_new_rating


@torch.no_grad()
def tournament(env,agent1,agent2,model_names,training_params,duplicate_decks):
    hero = model_names[1]
    villain = model_names[0]
    hero_river = defaultdict(lambda:[])
    villain_river = defaultdict(lambda:[])
    hero_dict = defaultdict(lambda:{'actions':[]})
    villain_dict = defaultdict(lambda:{'actions':[]})
    hero_dict['river'] = hero_river
    villain_dict['river'] = villain_river
    model_dict = {hero:hero_dict,villain:villain_dict}
    agent_performance = {
        villain: {'SB':0,'BB':0},
        hero: {'SB':0,'BB':0}
    }
    for e in range(training_params['epochs']):
        sys.stdout.write('\r')
        if e % 2 == 0:
            agent_positions = {'SB':agent1,'BB':agent2}
            agent_loc = {'SB':villain,'BB':hero}
        else:
            agent_positions = {'SB':agent2,'BB':agent1}
            agent_loc = {'SB':hero,'BB':villain}
        state,obs,done,action_mask,betsize_mask = env.reset(duplicate_decks[e])
        while not done:
            actor_outputs = agent_positions[env.current_player](state,action_mask,betsize_mask)
            street = pdt.Globals.STREET_DICT[state[:,-1,env.state_mapping['street']][0]]
            if street == pdt.StreetStrs.RIVER:
                if agent_loc[env.current_player] == hero:
                    hero_handstrength = hardcode_handstrength(torch.from_numpy(obs[:,-1,env.obs_mapping['hand_board']][:,None,:]))[0][0][0]
                    villain_handstrength = hardcode_handstrength(torch.from_numpy(obs[:,-1,env.obs_mapping['villain_board']][:,None,:]))[0][0][0]
                    model_dict[hero][street]['counts'].append(bin_by_handstrength(hero_handstrength))
                    model_dict[villain][street]['counts'].append(bin_by_handstrength(villain_handstrength))
                    hero_category = bin_by_handstrength(hero_handstrength)
                    model_dict[hero][street][hero_category].append(actor_outputs['action_category'])
                else:
                    villain_handstrength = hardcode_handstrength(torch.from_numpy(obs[:,-1,env.obs_mapping['hand_board']][:,None,:]))[0][0][0]
                    hero_handstrength = hardcode_handstrength(torch.from_numpy(obs[:,-1,env.obs_mapping['villain_board']][:,None,:]))[0][0][0]
                    model_dict[hero][street]['counts'].append(bin_by_handstrength(hero_handstrength))
                    model_dict[villain][street]['counts'].append(bin_by_handstrength(villain_handstrength))
                    villain_category = bin_by_handstrength(villain_handstrength)
                    model_dict[villain][street][villain_category].append(actor_outputs['action_category'])
            else:
                model_dict[agent_loc[env.current_player]][street]['actions'].append(actor_outputs['action_category'])
            state,obs,done,action_mask,betsize_mask = env.step(actor_outputs)
        rewards = env.player_rewards()
        agent_performance[agent_loc['SB']]['SB'] += rewards['SB']
        agent_performance[agent_loc['BB']]['BB'] += rewards['BB']
        sys.stdout.write("[%-60s] %d%%" % ('='*(60*(e+1)//training_params['epochs']), (100*(e+1)//training_params['epochs'])))
        sys.stdout.flush()
        sys.stdout.write(", epoch %d"% (e+1))
        sys.stdout.flush()
    return agent_performance,model_dict

def count_actions(values):
    raises = 0
    folds = 0
    calls = 0
    checks = 0
    bets = 0
    total_vals = 0
    for val in values:
        total_vals += 1
        if val == 0:
            checks += 1
        elif val == 1:
            folds += 1
        elif val == 2:
            calls += 1
        elif val == 3:
            bets += 1
        else:
            raises += 1
    return [checks/total_vals,folds/total_vals,calls/total_vals,bets/total_vals,raises/total_vals,total_vals]

def print_stats(stats):
    for model,data in stats.items():
        print(model)
        table = PrettyTable(['Street','Hand Category','Check','Fold','Call','Bet','Raise','Hand Counts'])
        for street in tuple(data.keys()):
            values = data[street]
            if street == pdt.StreetStrs.RIVER:
                counts = values['counts']
                uniques,freqs = np.unique(counts,return_counts=True)
                for category in range(9):
                    category_occurances = 0 if category not in uniques else freqs[uniques == category][0]
                    category_vals = values[category]
                    if category_vals:
                        checks,folds,calls,bets,raises,total_vals = count_actions(category_vals)
                        table.add_row([street,category,checks,folds,calls,bets,raises,category_occurances])
            else:
                checks,folds,calls,bets,raises,total_vals = count_actions(values['actions'])
                table.add_row([street,-1,checks,folds,calls,bets,raises,total_vals])
        print(table)

def eval_latest(env,seed,nS,nA,nB,training_params,network_params,duplicate_decks):
    device = network_params['device']
    weight_paths = load_paths(training_params['actor_path'])
    model_names = list(weight_paths.keys())
    model_names.sort(key=lambda l: int(grep("\d+", l)))
    latest_actor = model_names[-1]
    latest_net = OmahaActor(seed,nS,nA,nB,network_params).to(device)
    load_weights(latest_net,weight_paths[latest_actor])
    # Build matchups
    last_n_models = min(len(model_names),3)
    matchups = [(latest_actor,model) for model in model_names[-last_n_models:-1]]
    # create array to store results
    result_array = np.zeros(len(matchups))
    data_row_dict = {model:i for i,model in enumerate(model_names[-last_n_models:-1])}
    for match in matchups:
        net2 = OmahaActor(seed,nS,nA,nB,network_params).to(device)
        net2_path = weight_paths[match[1]]
        load_weights(net2,net2_path)
        results,stats = tournament(env,latest_net,net2,match,training_params,duplicate_decks)
        result_array[data_row_dict[match[1]]] = (results[match[0]]['SB'] + results[match[0]]['BB']) - (results[match[1]]['SB'] + results[match[1]]['BB'])
        print_stats(stats)
    # Create Results Table
    table = PrettyTable(["Model Name", *model_names[-last_n_models:-1]])
    table.add_row([latest_actor,*result_array])
    print(table)

def generate_duplicate_decks(epochs):
    """Positions alternate everyhand, so it is sufficient to just keep the deck the same and append it"""
    assert epochs % 2 == 0
    deck = deque(maxlen=52)
    for i in range(pdt.RANKS.LOW,pdt.RANKS.HIGH):
        for j in range(pdt.SUITS.LOW,pdt.SUITS.HIGH):
            deck.append([i,j])
    duplicate_decks = []
    for i in range(epochs // 2):
        shuffle(deck)
        duplicate_decks.append(copy.deepcopy(deck))
        duplicate_decks.append(copy.deepcopy(deck))
        # reversed_deck[-1],reversed_deck[-5] = reversed_deck[-5],reversed_deck[-1]
        # reversed_deck[-2],reversed_deck[-6] = reversed_deck[-6],reversed_deck[-2]
        # reversed_deck[-3],reversed_deck[-7] = reversed_deck[-7],reversed_deck[-3]
        # reversed_deck[-4],reversed_deck[-8] = reversed_deck[-8],reversed_deck[-4]
    return duplicate_decks

def run_tournament(actor,villain,model_names,params):
    config = Config()
    game_object = pdt.Globals.GameTypeDict[pdt.GameTypes.OMAHAHI]

    env_params = {
        'game':pdt.GameTypes.OMAHAHI,
        'betsizes': game_object.rule_params['betsizes'],
        'bet_type': game_object.rule_params['bettype'],
        'n_players': 2,
        'pot':game_object.state_params['pot'],
        'stacksize': game_object.state_params['stacksize'],
        'cards_per_player': game_object.state_params['cards_per_player'],
        'starting_street': game_object.starting_street,
        'global_mapping':config.global_mapping,
        'state_mapping':config.state_mapping,
        'obs_mapping':config.obs_mapping,
        'shuffle':False
    }
    print(f'Environment Parameters: Starting street: {env_params["starting_street"]},\
        Stacksize: {env_params["stacksize"]},\
        Pot: {env_params["pot"]},\
        Bettype: {env_params["bet_type"]},\
        Betsizes: {env_params["betsizes"]}')
    env = Poker(env_params)
    duplicate_decks = generate_duplicate_decks(params['epochs'])
    return tournament(env,actor,villain,model_names,params,duplicate_decks)

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
    parser.add_argument('--baseline','-b',
                        dest='baseline',
                        default='hardcoded',
                        type=str,
                        metavar=f'[hardcoded,baseline]',
                        help='which baseline to eval against')

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
        'pot':game_object.state_params['pot'],
        'stacksize': game_object.state_params['stacksize'],
        'cards_per_player': game_object.state_params['cards_per_player'],
        'starting_street': game_object.starting_street,
        'global_mapping':config.global_mapping,
        'state_mapping':config.state_mapping,
        'obs_mapping':config.obs_mapping,
        'shuffle':False
    }
    print(f'Environment Parameters: Starting street: {env_params["starting_street"]},\
        Stacksize: {env_params["stacksize"]},\
        Pot: {env_params["pot"]},\
        Bettype: {env_params["bet_type"]},\
        Betsizes: {env_params["betsizes"]}')
    env = Poker(env_params)
    duplicate_decks = generate_duplicate_decks(args.epochs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_params = config.training_params
    training_params['epochs'] = args.epochs
    training_params['save_dir']:os.path.join(os.getcwd(),'checkpoints/training_run')

    network_params            = config.network_params
    network_params['device']  = device

    nS = env.state_space
    nO = env.observation_space
    nA = env.action_space
    nB = env.betsize_space
    model_name = 'OmahaActorFinal' if args.network_type == 'dual' else 'OmahaCombinedFinal'
    print(f'Environment: State Space {nS}, Obs Space {nO}, Action Space {nA}, Betsize Space {nB}\n')
    seed = 154

    if args.network_type == 'dual':
        trained_model = OmahaActor(seed,nS,nA,nB,network_params).to(device)
    else:
        trained_model = CombinedNet(seed,nS,nA,nB,network_params).to(device)
    if args.tourney == 'latest':
        """Takes the latest network weights and evals vs all the previous ones or the last N"""
        # load all file paths
        eval_latest(env,seed,nS,nA,nB,training_params,network_params,duplicate_decks)
    elif args.tourney == 'roundrobin':
        """Runs all saved weights (in training_run folder) against each other in a round robin"""
        # load all file paths
        weight_paths = load_paths(training_params['actor_path'])
        print('weight_paths',weight_paths)
        # all combinations
        model_names = list(weight_paths.keys())
        model_names.sort(key=lambda l: int(grep("\d+", l)))
        matchups = list(combinations(model_names,2))
        # create array to store results
        result_array = np.zeros((len(model_names),len(model_names)))
        data_row_dict = {model:i for i,model in enumerate(model_names)}
        for match in matchups:
            net1 = OmahaActor(seed,nS,nA,nB,network_params).to(device)
            net2 = OmahaActor(seed,nS,nA,nB,network_params).to(device)
            net1_path = weight_paths[match[0]]
            net2_path = weight_paths[match[1]]
            load_weights(net1,net1_path)
            load_weights(net2,net2_path)
            results,stats = tournament(env,net1,net2,match,training_params,duplicate_decks)
            result_array[data_row_dict[match[0]],data_row_dict[match[1]]] = results[match[0]]['SB'] + results[match[0]]['BB']
            result_array[data_row_dict[match[1]],data_row_dict[match[0]]] = results[match[1]]['SB'] + results[match[1]]['BB']
        # Create Results Table
        table = PrettyTable(["Model Name", *model_names])
        for i,model in enumerate(model_names):
            row = list(result_array[i])
            row[i] = 'x'
            table.add_row([model,*row])
        print(table)
    else:
        print(f'Evaluating {model_name}, from {os.path.join(training_params["actor_path"],model_name)}')
        load_weights(trained_model,os.path.join(training_params['actor_path'],model_name))
        if args.baseline == 'hardcoded':
            baseline_evaluation = BetAgent()
        else:
            baseline_evaluation = OmahaActor(seed,nS,nA,nB,network_params).to(device)
            baseline_path = return_latest_baseline_path(config.baseline_path)
            print('baseline_path',baseline_path)
            load_weights(baseline_evaluation,baseline_path)
            # Get latest baseline
        model_names = ['baseline_evaluation','trained_model']
        results,stats = tournament(env,baseline_evaluation,trained_model,model_names,training_params,duplicate_decks)
        print(results)
        print_stats(stats)

        print(f"{model_names[0]}: {results[model_names[0]]['SB'] + results[model_names[0]]['BB']}")
        print(f"{model_names[1]}: {results[model_names[1]]['SB'] + results[model_names[1]]['BB']}")
        toc = time.time()
        print(f'Tournament completed in {(toc-tic)/60} minutes')