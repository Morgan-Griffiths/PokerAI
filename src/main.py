
import time
import torch.multiprocessing as mp
import torch
import os
import copy

from train import train,generate_trajectories,learning_update
from poker.config import Config
import poker.datatypes as pdt
from poker.env import Poker
from db import MongoDB
from models.network_config import NetworkConfig,CriticType
from models.networks import OmahaActor,OmahaQCritic
from models.model_utils import update_weights
from agents.agent import return_agent
from utils.utils import unpack_shared_dict

from torch import optim

if __name__ == "__main__":
    import argparse

    print(f"Number of processors: {mp.cpu_count()}")
    print(f'Number of GPUs: {torch.cuda.device_count()}')
    tic = time.time()

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
    env = Poker(env_params)

    nS = env.state_space
    nA = env.action_space
    nB = env.betsize_space
    seed = 1235
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu1 = 'cuda:0'
    gpu2 = 'cuda:1'

    network_params = {
        'game':pdt.GameTypes.OMAHAHI,
        'maxlen':config.maxlen,
        'state_mapping':config.state_mapping,
        'embedding_size':128,
        'device':device,
        'frozen_layer_path':os.path.join(os.getcwd(),'checkpoints/regression/PartialHandRegression')
    }
    # critic_network_params = copy.deepcopy(network_params)
    # critic_network_params['device'] = gpu2
    training_params = {
        'training_epochs':50,
        'epochs':25,
        'training_round':0,
        'game':'Omaha',
        'id':0
    }

    actor = OmahaActor(seed,nS,nA,nB,network_params).to(device)
    critic = OmahaQCritic(seed,nS,nA,nB,network_params).to(device)
    # preload the hand board analyzer
    actor,critic = update_weights([actor,critic],network_params['frozen_layer_path'])
    actor_optimizer = optim.Adam(actor.parameters(), lr=config.agent_params['actor_lr'],weight_decay=config.agent_params['L2'])
    critic_optimizer = optim.Adam(critic.parameters(), lr=config.agent_params['critic_lr'])

    learning_params = {
        'training_round':0,
        'gradient_clip':config.agent_params['CLIP_NORM'],
        'actor_optimizer':actor_optimizer,
        'critic_optimizer':critic_optimizer,
        'path': os.path.join(os.getcwd(),'checkpoints/RL/omaha_hi'),
        'device':device,
        'gpu1':gpu1,
        'gpu2':gpu2
    }
    mp.set_start_method('spawn')
    # generate trajectories and desposit in mongoDB
    mongo = MongoDB()
    mongo.clean_db()
    mongo.close()
    # training loop
    actor.share_memory()
    critic.share_memory()
    processes = []
    num_processes = min(mp.cpu_count(),1)
    print(f"Number of processors used: {num_processes}")
    tic = time.time()
    for id in range(num_processes): # No. of processes
        p = mp.Process(target=train, args=(env,actor,critic,training_params,learning_params,id))
        p.start()
        processes.append(p)
    for p in processes: 
        p.join()
    # save weights
    path = learning_params['path']
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.mkdir(directory)
    torch.save(actor.state_dict(), path + '_actor')
    torch.save(critic.state_dict(), path + '_critic')
    toc = time.time()
    print(f'Training completed in {(toc-tic)/60} minutes')