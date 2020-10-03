
from torch import optim
import torch.nn.functional as F
import torch
import os
from pymongo import MongoClient
import numpy as np
import sys

from db import MongoDB
from poker_env.config import Config
import poker_env.datatypes as pdt
from poker_env.env import Poker
from utils.data_loaders import return_trajectoryloader
from train import generate_trajectories,dual_learning_update,combined_learning_update
from models.networks import CombinedNet,OmahaActor,OmahaQCritic,OmahaObsQCritic,OmahaBatchActor,OmahaBatchObsQCritic
from models.model_updates import update_combined,update_actor_critic,update_critic,update_actor,update_critic_batch
from models.model_utils import scale_rewards,soft_update,hard_update,return_value_mask

def eval_batch_critic(critic,target_critic,params):
    device = params['device']
    critic_optimizer = params['critic_optimizer']
    actor_optimizer = params['actor_optimizer']
    query = {'training_round':params['training_round']}
    projection = {'state':1,'betsize_mask':1,'action_mask':1,'action':1,'reward':1,'_id':0}
    client = MongoClient('localhost', 27017,maxPoolSize=10000)
    db = client['poker']
    data = db['game_data'].find(query,projection)
    trainloader = return_trajectoryloader(data)
    losses = []
    for i,data in enumerate(trainloader,1):
        loss = update_critic_batch(critic,target_critic,data,params)
        losses.append(loss)
    print(losses)

def eval_critic(critic,params):
    query = {'training_round':0}
    projection = {'obs':1,'state':1,'betsize_mask':1,'action_mask':1,'action':1,'reward':1,'_id':0}
    client = MongoClient('localhost', 27017,maxPoolSize=10000)
    db = client['poker']
    data = list(db['game_data'].find(query,projection))
    print(f'Number of data points {len(data)}')
    for i in range(params['learning_rounds']):
        losses = []
        for j,poker_round in enumerate(data,1):
            sys.stdout.write('\r')
            critic_loss = update_critic(poker_round,critic,params)
            losses.append(critic_loss)
            sys.stdout.write("[%-60s] %d%%" % ('='*(60*(j)//len(data)), (100*(j)//len(data))))
            sys.stdout.flush()
            sys.stdout.write(f", round {(j):.2f}")
            sys.stdout.flush()
        print(f'Training Round {i}, critic loss {sum(losses)}')
    del data

def eval_actor(actor,target_actor,target_critic,params):
    query = {'training_round':0}
    projection = {'obs':1,'state':1,'betsize_mask':1,'action_mask':1,'action':1,'reward':1,'_id':0}
    client = MongoClient('localhost', 27017,maxPoolSize=10000)
    db = client['poker']
    data = list(db['game_data'].find(query,projection))
    print(f'Number of data points {len(data)}')
    for i in range(params['learning_rounds']):
        for j,poker_round in enumerate(data,1):
            # sys.stdout.write('\r')
            update_actor(poker_round,actor,target_actor,target_critic,params)
            # sys.stdout.write("[%-60s] %d%%" % ('='*(60*(j)//len(data)), (100*(j)//len(data))))
            # sys.stdout.flush()
            # sys.stdout.write(f", round {(j):.2f}")
            # sys.stdout.flush()
            print(f'Training Round {j}')
            break
    del data

def eval_network_updates(actor,critic,target_actor,target_critic,params):
    critic_optimizer = params['critic_optimizer']  
    actor_optimizer = params['actor_optimizer']  
    device = params['device']
    query = {'training_round':0}
    projection = {'obs':1,'state':1,'betsize_mask':1,'action_mask':1,'action':1,'reward':1,'_id':0}
    client = MongoClient('localhost', 27017,maxPoolSize=10000)
    db = client['poker']
    data = list(db['game_data'].find(query,projection))
    print(f'Number of data points {len(data)}')
    for i in range(params['learning_rounds']):
        losses = []
        policy_losses = []
        for j,poker_round in enumerate(data,1):
            sys.stdout.write('\r')
            critic_loss,policy_loss = update_actor_critic(poker_round,critic,target_critic,actor,target_actor,params)
            losses.append(critic_loss)   
            policy_losses.append(policy_loss)
            sys.stdout.write("[%-60s] %d%%" % ('='*(60*(j)//len(data)), (100*(j)//len(data))))
            sys.stdout.flush()
            sys.stdout.write(f", round {(j):.2f}")
            sys.stdout.flush()
        print(f'\nTraining Round {i}, critic loss {sum(losses)}, policy loss {sum(policy_losses)}')
    del data

def eval_combined_updates(model,params):
    query = {'training_round':0}
    projection = {'state':1,'betsize_mask':1,'action_mask':1,'action':1,'reward':1,'_id':0}
    client = MongoClient('localhost', 27017,maxPoolSize=10000)
    db = client['poker']
    data = list(db['game_data'].find(query,projection))
    print(f'Number of data points {len(data)}')
    for i in range(params['learning_rounds']):
        losses = []
        policy_losses = []
        for j,poker_round in enumerate(data,1):
            sys.stdout.write('\r')
            critic_loss,policy_loss = update_combined(poker_round,model,params)
            losses.append(critic_loss) 
            policy_losses.append(policy_loss)
            sys.stdout.write("[%-60s] %d%%" % ('='*(60*(j)//len(data)), (100*(j)//len(data))))
            sys.stdout.flush()
            sys.stdout.write(f", round {(j):.2f}")
            sys.stdout.flush()
        print(f'Training Round {i}, critic loss {sum(losses)}, policy loss {sum(policy_losses)}')
    del data

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=
        """
        Evaluates learning methods on static data.
        """)

    parser.add_argument('--network','-n',
                        dest='network_type',
                        default='dual',
                        metavar="['combined','dual']",
                        type=str,
                        help='eval actor/critic or combined networks')
    parser.add_argument('--eval','-e',
                        dest='eval',
                        default='actor_critic',
                        metavar="['actor_critic','actor','critic]",
                        type=str,
                        help='In dual networks, whether to eval both or one at a time')

    args = parser.parse_args()

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
    network_params = config.network_params
    network_params['device'] = device
    training_params = {
        'training_epochs':50,
        'generate_epochs':10,
        'training_round':0,
        'game':pdt.GameTypes.OMAHAHI,
        'id':0
    }
    learning_params = {
        'training_round':0,
        'gradient_clip':config.agent_params['CLIP_NORM'],
        'path': os.path.join(os.getcwd(),'checkpoints'),
        'learning_rounds':100,
        'device':device,
        'gpu1':gpu1,
        'gpu2':gpu2,
        'min_reward':-env_params['stacksize'],
        'max_reward':env_params['pot']+env_params['stacksize']
    }

    print(f'Environment Parameters: Starting street: {env_params["starting_street"]},\
        Stacksize: {env_params["stacksize"]},\
        Pot: {env_params["pot"]},\
        Bettype: {env_params["bet_type"]},\
        Betsizes: {env_params["betsizes"]}')
    print(f'Evaluating {args.network_type}')

    # Clean mongo
    mongo = MongoDB()
    mongo.clean_db()
    mongo.close()

    if args.network_type == 'combined':
        # Instantiate network
        alphaPoker = CombinedNet(seed,nS,nA,nB,network_params).to(device)
        alphaPoker_optimizer = optim.Adam(alphaPoker.parameters(), lr=config.agent_params['critic_lr'])
        learning_params['model_optimizer'] = alphaPoker_optimizer
        # Gen trajectories
        generate_trajectories(env,alphaPoker,training_params,id=0)
        eval_combined_updates(alphaPoker,learning_params)
    else:
        local_actor = OmahaActor(seed,nS,nA,nB,network_params).to(device)
        target_actor = OmahaActor(seed,nS,nA,nB,network_params).to(device)
        local_critic = OmahaObsQCritic(seed,nS,nA,nB,network_params).to(device)
        target_critic = OmahaObsQCritic(seed,nS,nA,nB,network_params).to(device)
        hard_update(target_actor,local_actor)
        hard_update(target_critic,local_critic)
        actor_optimizer = optim.Adam(local_actor.parameters(), lr=config.agent_params['actor_lr'],weight_decay=config.agent_params['L2'])
        critic_optimizer = optim.Adam(local_critic.parameters(), lr=config.agent_params['critic_lr'])
        learning_params['actor_optimizer'] = actor_optimizer
        learning_params['critic_optimizer'] = critic_optimizer

        # Gen trajectories
        generate_trajectories(env,local_actor,local_critic,training_params,id=0)

        # Eval learning models
        if args.eval == 'actor':
            eval_actor(local_actor,target_actor,target_critic,learning_params)
        elif args.eval == 'critic':
            eval_critic(local_critic,learning_params)
            # eval_batch_critic(local_critic,target_critic,learning_params)
        else:
            eval_network_updates(local_actor,local_critic,target_actor,target_critic,learning_params)