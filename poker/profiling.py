from torch import optim
import torch.nn.functional as F
import torch
import torch.autograd.profiler as profiler
import os
from pymongo import MongoClient
import numpy as np
import sys
import time

from db import MongoDB
from poker_env.config import Config
import poker_env.datatypes as pdt
from poker_env.env import Poker
from utils.data_loaders import return_trajectoryloader
from train import generate_trajectories,dual_learning_update,combined_learning_update
from models.networks import CombinedNet,OmahaActor,OmahaQCritic,OmahaObsQCritic,OmahaBatchActor,OmahaBatchObsQCritic
from models.model_updates import update_combined,update_actor_critic,update_critic,update_actor,update_critic_batch,update_actor_batch,update_actor_critic_batch
from models.model_utils import scale_rewards,soft_update,hard_update,return_value_mask,copy_weights



if __name__ == "__main__":
    import argparse

    config = Config()
    game_object = pdt.Globals.GameTypeDict[pdt.GameTypes.OMAHAHI]

    env_params = {
        'game':pdt.GameTypes.OMAHAHI,
        'betsizes': game_object.rule_params['betsizes'],
        'bet_type': game_object.rule_params['bettype'],
        'n_players': 2,
        'pot':0,
        'stacksize': game_object.state_params['stacksize'],
        'cards_per_player': game_object.state_params['cards_per_player'],
        'starting_street': pdt.Street.PREFLOP, #game_object.starting_street,
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
    network_params = config.network_params
    network_params['device'] = device
    training_params = {
        'training_epochs':50,
        'generate_epochs':10,
        'training_round':0,
        'game':pdt.GameTypes.OMAHAHI,
        'id':0
    }
    # learning_params = {
    #     'training_round':0,
    #     'gradient_clip':config.agent_params['CLIP_NORM'],
    #     'path': os.path.join(os.getcwd(),'checkpoints'),
    #     'learning_rounds':args.epochs,
    #     'device':device,
    #     'gpu1':gpu1,
    #     'gpu2':gpu2,
    #     'min_reward':-env_params['stacksize'],
    #     'max_reward':env_params['pot']+env_params['stacksize']
    # }

    print(f'Environment Parameters: Starting street: {env_params["starting_street"]},\
        Stacksize: {env_params["stacksize"]},\
        Pot: {env_params["pot"]},\
        Bettype: {env_params["bet_type"]},\
        Betsizes: {env_params["betsizes"]}')
    # print(f'Evaluating {args.network_type}')

    actor = OmahaActor(seed,nS,nA,nB,network_params).to(device)
    critic = OmahaBatchObsQCritic(seed,nS,nA,nB,network_params).to(device)
    # Clean mongo
    # mongo = MongoDB()
    # mongo.clean_db()
    # mongo.close()
    # Check env steps

    # test generate
    tic = time.time()
    with profiler.profile(record_shapes=True) as prof:
        generate_trajectories(env,actor,critic,training_params,id=0)
    print(f'Computation took {time.time() - tic} seconds')
    print(prof)

    # test env
    # tic = time.time()
    # with profiler.profile(record_shapes=True) as prof:
    #     with torch.no_grad():
    #         for i in range(100):
    #             state,obs,done,action_mask,betsize_mask = env.reset()
    #             while not done:
    #                 actor_outputs = actor(state,action_mask,betsize_mask)
    #                 state,obs,done,action_mask,betsize_mask = env.step(actor_outputs)
    # print(f'Computation took {time.time() - tic} seconds')
    # print(prof)