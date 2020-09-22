
import time
import torch.multiprocessing as mp
import torch
import os

from train import train,generate_trajectories,learning_update
from poker_env.config import Config
import poker_env.datatypes as pdt
from poker_env.env import Poker
from db import MongoDB
from models.network_config import NetworkConfig,CriticType
from models.networks import OmahaActor,OmahaQCritic
from utils.utils import unpack_shared_dict

from torch import optim

if __name__ == "__main__":
    import argparse

    print("Number of processors: ", mp.cpu_count())
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
    print(f'Environment Parameters: Starting street: {env_params["starting_street"]},\
        Stacksize: {env_params["stacksize"]},\
        Pot: {env_params["pot"]},\
        Bettype: {env_params["bet_type"]},\
        Betsizes: {env_params["betsizes"]}')
    env = Poker(env_params)

    nS = env.state_space
    nA = env.action_space
    nB = env.betsize_space
    seed = 1235

    network_params = {
        'game':pdt.GameTypes.OMAHAHI,
        'maxlen':config.maxlen,
        'state_mapping':config.state_mapping,
        'embedding_size':128
    }
    training_params = {
        'training_epochs':1,
        'epochs':1,
        'training_round':0,
        'game':'Omaha',
        'id':0
    }

    actor = OmahaActor(seed,nS,nA,nB,network_params)
    critic = OmahaQCritic(seed,nS,nA,nB,network_params)
    actor_optimizer = optim.Adam(actor.parameters(), lr=config.agent_params['actor_lr'],weight_decay=config.agent_params['L2'])
    critic_optimizer = optim.Adam(critic.parameters(), lr=config.agent_params['critic_lr'])

    learning_params = {
        'gradient_clip':config.agent_params['CLIP_NORM'],
        'actor_optimizer':actor_optimizer,
        'critic_optimizer':critic_optimizer,
        'path': os.path.join(os.getcwd(),'checkpoints'),
        'learning_rounds':1
    }
    # generate trajectories and desposit in mongoDB
    mongo = MongoDB()
    mongo.clean_db()
    mongo.close()
    # training loop
    actor.share_memory()#.to(device)
    critic.share_memory()#.to(device)
    processes = []
    num_processes = mp.cpu_count()
    # for debugging
    # generate_trajectories(env,actor,training_params,id=0)
    # actor,critic,learning_params = learning_update(actor,critic,learning_params)
    train(env,actor,critic,training_params,learning_params,id=0)
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
    torch.save(actor.state_dict(), os.path.join(path,'RL_actor'))
    torch.save(critic.state_dict(), os.path.join(path,'RL_critic'))
