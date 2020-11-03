from torch import optim
import torch.nn.functional as F
import torch
import torch.autograd.profiler as profiler
from torch.optim.lr_scheduler import MultiStepLR,StepLR
import os
from pymongo import MongoClient
import numpy as np
import sys
import matplotlib.pyplot as plt
import time

from db import MongoDB
from poker_env.config import Config
import poker_env.datatypes as pdt
from poker_env.env import Poker
from utils.data_loaders import return_trajectoryloader
from train import generate_trajectories,dual_learning_update,train_dual
from models.networks import CombinedNet,OmahaActor,OmahaQCritic,OmahaObsQCritic,OmahaBatchActor,OmahaBatchObsQCritic
from models.model_updates import update_combined,update_actor_critic,update_critic,update_actor,update_critic_batch,update_actor_batch,update_actor_critic_batch
from models.model_utils import scale_rewards,soft_update,hard_update,return_value_mask,copy_weights



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=
        """
        Profiling all the function calls
        """)

    parser.add_argument('--function','-f',
                        dest='function',
                        default='dual',
                        metavar="['generate','learn','train','env']",
                        type=str,
                        help='which function to call')

    args = parser.parse_args()

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
        'training_epochs':1,
        'generate_epochs':10,
        'training_round':0,
        'game':'Omaha',
        'id':0,
        'save_every':max(100 // 4,1),
        'save_dir':os.path.join(os.getcwd(),'checkpoints/training_run'),
        'actor_path':config.agent_params['actor_path'],
        'critic_path':config.agent_params['critic_path'],
        'baseline_path':config.baseline_path
    }
    learning_params = {
        'training_round':0,
        'gradient_clip':config.agent_params['CLIP_NORM'],
        'path': os.path.join(os.getcwd(),'checkpoints'),
        'learning_rounds':1,
        'device':device,
        'gpu1':gpu1,
        'gpu2':gpu2,
        'min_reward':-env_params['stacksize'],
        'max_reward':env_params['pot']+env_params['stacksize']
    }
    validation_params = {
        'epochs':5000,
        'koth':False
    }

    print(f'Environment Parameters: Starting street: {env_params["starting_street"]},\
        Stacksize: {env_params["stacksize"]},\
        Pot: {env_params["pot"]},\
        Bettype: {env_params["bet_type"]},\
        Betsizes: {env_params["betsizes"]}')
    # print(f'Evaluating {args.network_type}')
    actor = OmahaActor(seed,nS,nA,nB,network_params).to(device)
    critic = OmahaBatchObsQCritic(seed,nS,nA,nB,network_params).to(device)
    target_actor = OmahaActor(seed,nS,nA,nB,network_params).to(device)
    target_critic = OmahaBatchObsQCritic(seed,nS,nA,nB,network_params).to(device)
    actor_optimizer = optim.Adam(actor.parameters(), lr=config.agent_params['actor_lr'],weight_decay=config.agent_params['L2'])
    critic_optimizer = optim.Adam(critic.parameters(), lr=config.agent_params['critic_lr'])
    actor_lrscheduler = StepLR(actor_optimizer, step_size=1, gamma=0.1)
    critic_lrscheduler = StepLR(critic_optimizer, step_size=1, gamma=0.1)
    learning_params['actor_optimizer'] = actor_optimizer
    learning_params['critic_optimizer'] = critic_optimizer
    learning_params['actor_lrscheduler'] = actor_lrscheduler
    learning_params['critic_lrscheduler'] = critic_lrscheduler
    # Clean mongo
    mongo = MongoDB()
    mongo.clean_db()
    mongo.close()
    print(args)
    times = []
    for i,val in enumerate([1,2,5,10,25,50]):
        print(f'Generating {val} samples')
        tic = time.time()
        training_params['generate'] = val
        train_dual(env,actor,critic,target_actor,target_critic,training_params,learning_params,network_params,validation_params,id=0)
        toc = time.time()
        print(f'{val} samples took {toc-tic} seconds')
        times.append(toc-tic)
        mongo = MongoDB()
        mongo.clean_db()
        mongo.close()
    plt.plot(times)
    plt.savefig(f'generate_times.png',bbox_inches='tight')
    # tic = time.time()
    # with profiler.profile(record_shapes=True) as prof:
    #     if args.function == 'train':
    #     # cProfile.run('train_dual(env,actor,critic,target_actor,target_critic,training_params,learning_params,network_params,validation_params,id=0)')
    #         train_dual(env,actor,critic,target_actor,target_critic,training_params,learning_params,network_params,validation_params,id=0)
    #     elif args.function == 'learn':
    #         dual_learning_update(actor,critic,target_actor,target_critic,learning_params)
    #     elif args.function == 'generate':
    #         generate_trajectories(env,actor,critic,training_params,id=0)
    #     else:
    #         with torch.no_grad():
    #             for i in range(100):
    #                 state,obs,done,action_mask,betsize_mask = env.reset()
    #                 while not done:
    #                     actor_outputs = actor(state,action_mask,betsize_mask)
    #                     state,obs,done,action_mask,betsize_mask = env.step(actor_outputs)
    # print(f'Computation took {time.time() - tic} seconds')
    # print(prof)