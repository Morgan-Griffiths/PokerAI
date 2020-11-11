import os
import poker_env.datatypes as pdt
import copy
import torch
import sys
import numpy as np
from pymongo import MongoClient
from pymongo import DESCENDING,ASCENDING
from collections import defaultdict
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.autograd.profiler as profiler
import datetime
import copy
import time
import logging
from torch import optim
from models.networks import OmahaActor,OmahaQCritic,OmahaObsQCritic,CombinedNet,BetAgent
from models.model_updates import update_actor_critic,update_combined,update_critic_batch,update_actor_critic_batch
from utils.data_loaders import return_trajectoryloader
from utils.utils import return_latest_training_model_path,return_latest_baseline_path
from models.model_utils import soft_update,load_weights,hard_update
from db import MongoDB
from poker_env.env import Poker

def load_villain(rank,network_params,baseline_path):
    baseline_path = return_latest_baseline_path(baseline_path)
    if baseline_path:
        seed = network_params['seed']
        nS = network_params['nS']
        nA = network_params['nA']
        nB = network_params['nB']
        villain = OmahaActor(seed,nS,nA,nB,network_params).to(rank)
        load_weights(villain,baseline_path)
    else:
        villain = BetAgent()
    return villain

def pad_state(state,maxlen):
    N = maxlen - state.shape[1]
    padding = np.zeros(N)
    return padded_state

def setup_world(rank,world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size,timeout=datetime.timedelta(0, 60))

def cleanup():
    dist.destroy_process_group()

@torch.no_grad()
def generate_vs_frozen(env,actor,critic,villain,training_params,rank):
    trajectories = defaultdict(lambda:[])
    for e in range(training_params['generate_epochs']):
        trajectory = defaultdict(lambda:{'states':[],'obs':[],'betsize_masks':[],'action_masks':[], 'actions':[],'action_category':[],'action_probs':[],'action_prob':[],'betsize':[],'rewards':[],'values':[]})
        state,obs,done,action_mask,betsize_mask = env.reset()
        cur_player = env.current_player
        if e % 2 == 0:
            actor_positions = {'SB':actor,'BB':villain}
            critic_positions = {'SB':critic,'BB':villain}
            agent_loc = {'SB':1,'BB':0}
        else:
            actor_positions = {'SB':villain,'BB':actor}
            critic_positions = {'SB':villain,'BB':critic}
            agent_loc = {'SB':0,'BB':1}
        if agent_loc[cur_player]:
            trajectory[cur_player]['states'].append(copy.copy(state))
            trajectory[cur_player]['obs'].append(copy.copy(obs))
            trajectory[cur_player]['action_masks'].append(copy.copy(action_mask))
            trajectory[cur_player]['betsize_masks'].append(copy.copy(betsize_mask))
        while not done:
            actor_outputs = actor_positions[env.current_player](state,action_mask,betsize_mask,target=True)
            if agent_loc[cur_player]:
                critic_outputs = critic_positions[env.current_player](obs)
                trajectory[cur_player]['values'].append(critic_outputs['value'])
                trajectory[cur_player]['actions'].append(actor_outputs['action'])
                trajectory[cur_player]['action_category'].append(actor_outputs['action_category'])
                trajectory[cur_player]['action_prob'].append(actor_outputs['action_prob'])
                trajectory[cur_player]['action_probs'].append(actor_outputs['action_probs'])
                trajectory[cur_player]['betsize'].append(actor_outputs['betsize'])
            state,obs,done,action_mask,betsize_mask = env.step(actor_outputs)
            cur_player = env.current_player
            if not done and agent_loc[cur_player]:
                trajectory[cur_player]['states'].append(state)
                trajectory[cur_player]['obs'].append(copy.copy(obs))
                trajectory[cur_player]['action_masks'].append(action_mask)
                trajectory[cur_player]['betsize_masks'].append(betsize_mask)
        assert len(trajectory[cur_player]['betsize']) == len(trajectory[cur_player]['betsize_masks'])
        rewards = env.player_rewards()
        for position in trajectory.keys():
            N = len(trajectory[position]['betsize_masks'])
            trajectory[position]['rewards'] = [rewards[position]] * N
            trajectories[position].append(trajectory[position])
    insert_data(trajectories,env.state_mapping,env.obs_mapping,training_params['training_round'],training_params['game'],rank,training_params['generate_epochs'])

@torch.no_grad()
def generate_trajectories(env,actor,critic,training_params,rank):
    """Generates full trajectories by playing against itself"""
    trajectories = defaultdict(lambda:[])
    for e in range(training_params['generate_epochs']):
        trajectory = defaultdict(lambda:{'states':[],'obs':[],'betsize_masks':[],'action_masks':[], 'actions':[],'action_category':[],'action_probs':[],'action_prob':[],'betsize':[],'rewards':[],'values':[]})
        state,obs,done,action_mask,betsize_mask = env.reset()
        cur_player = env.current_player
        trajectory[cur_player]['states'].append(copy.copy(state))
        trajectory[cur_player]['obs'].append(copy.copy(obs))
        trajectory[cur_player]['action_masks'].append(copy.copy(action_mask))
        trajectory[cur_player]['betsize_masks'].append(copy.copy(betsize_mask))
        while not done:
            actor_outputs = actor(state,action_mask,betsize_mask,target=True)
            critic_outputs = critic(obs)
            trajectory[cur_player]['values'].append(critic_outputs['value'])
            trajectory[cur_player]['actions'].append(actor_outputs['action'])
            trajectory[cur_player]['action_category'].append(actor_outputs['action_category'])
            trajectory[cur_player]['action_prob'].append(actor_outputs['action_prob'])
            trajectory[cur_player]['action_probs'].append(actor_outputs['action_probs'])
            trajectory[cur_player]['betsize'].append(actor_outputs['betsize'])
            state,obs,done,action_mask,betsize_mask = env.step(actor_outputs)
            cur_player = env.current_player
            if not done:
                trajectory[cur_player]['states'].append(state)
                trajectory[cur_player]['obs'].append(copy.copy(obs))
                trajectory[cur_player]['action_masks'].append(action_mask)
                trajectory[cur_player]['betsize_masks'].append(betsize_mask)
        assert len(trajectory[cur_player]['betsize']) == len(trajectory[cur_player]['betsize_masks'])
        rewards = env.player_rewards()
        for position in trajectory.keys():
            N = len(trajectory[position]['betsize_masks'])
            trajectory[position]['rewards'] = [rewards[position]] * N
            trajectories[position].append(trajectory[position])
    insert_data(trajectories,env.state_mapping,env.obs_mapping,training_params['training_round'],training_params['game'],rank,training_params['generate_epochs'])


def insert_data(training_data:dict,mapping:dict,obs_mapping,training_round:int,gametype:str,rank:int,epochs:int):
    """
    takes trajectories and inserts them into db for data analysis and learning.
    """
    client = MongoClient('localhost', 27017,maxPoolSize=10000)
    db = client['poker']
    keys = training_data.keys()
    positions = [position for position in keys if position in ['SB','BB']]   
    for position in positions:
        for i,poker_round in enumerate(training_data[position]):
            states = poker_round['states']
            observations = poker_round['obs']
            actions = poker_round['actions']
            action_prob = poker_round['action_prob']
            action_probs = poker_round['action_probs']
            action_categories = poker_round['action_category']
            betsize_masks = poker_round['betsize_masks']
            action_masks = poker_round['action_masks']
            rewards = poker_round['rewards']
            betsizes = poker_round['betsize']
            values = poker_round['values']
            assert(isinstance(rewards,list))
            assert(isinstance(actions,list))
            assert(isinstance(action_prob,list))
            assert(isinstance(action_probs,list))
            assert(isinstance(observations,list))
            assert(isinstance(states,list))
            for step,state in enumerate(states):
                state_json = {
                    'rank':rank,
                    'training_round':training_round,
                    'poker_round':i + (rank * epochs),
                    'state':state.tolist(),
                    'obs':observations[step].tolist(),
                    'action_probs':action_probs[step].tolist(),
                    'action_prob':action_prob[step].tolist(),
                    'action':actions[step],
                    'action_category':action_categories[step],
                    'betsize_mask':betsize_masks[step].tolist(),
                    'action_mask':action_masks[step].tolist(),
                    'betsize':betsizes[step],
                    'reward':rewards[step],
                    'values':values[step].tolist()
                }
                db['game_data'].insert_one(state_json)
    client.close()

def combined_learning_update(model,params):
    model.train()
    query = {'training_round':params['training_round']}
    projection = {'state':1,'betsize_mask':1,'action_mask':1,'action':1,'reward':1,'_id':0}
    client = MongoClient('localhost', 27017,maxPoolSize=10000)
    db = client['poker']
    data = list(db['game_data'].find(query,projection))
    # trainloader = return_trajectoryloader(data)
    # loss_dict = defaultdict(lambda:None)
    for i in range(params['learning_rounds']):
        losses = []
        policy_losses = []
        for poker_round in data:
            critic_loss,policy_loss = update_combined(poker_round,model,params)
            losses.append(critic_loss)
            policy_losses.append(policy_loss)
        print(f'Training Round {i}, critic loss {sum(losses)}, policy loss {sum(policy_losses)}')
    del data
    mongo.close()
    return model,params
    
def dual_learning_update(rank,actor,critic,target_actor,target_critic,params,validation_params):
    query = {'training_round':params['training_round']}
    projection = {'obs':1,'state':1,'betsize_mask':1,'action_mask':1,'action':1,'reward':1,'_id':0}
    client = MongoClient('localhost', 27017,maxPoolSize=10000)
    db = client['poker']
    while True:
        if db.game_data.count_documents({'training_round':params['training_round'],'rank':0}) > 0 and db.game_data.count_documents({'training_round':params['training_round'],'rank':1}) > 0:
            break
    count = db.game_data.count_documents({'training_round':params['training_round']}) // 2
    if rank == 0:
        data = db['game_data'].find(query,projection).sort('_id',ASCENDING).limit(count)
    else:
        data = db['game_data'].find(query,projection).sort('_id',DESCENDING).limit(count)
    for i in range(params['learning_rounds']):
        for poker_round in data:
            update_actor_critic(poker_round,critic,target_critic,actor,target_actor,params)
        soft_update(critic,target_critic,params['device'])
        soft_update(actor,target_actor,params['device'])
    client.close()

def batch_learning_update(rank,actor,critic,target_actor,target_critic,params):
    mongo = MongoDB()
    query = {'training_round':params['training_round'],'rank':rank}
    projection = {'obs':1,'state':1,'betsize_mask':1,'action_mask':1,'action':1,'reward':1,'_id':0}
    db_data = mongo.get_data(query,projection)
    trainloader = return_trajectoryloader(db_data)
    for _ in range(params['learning_rounds']):
        losses = []
        for i,data in enumerate(trainloader):
            critic_loss = update_actor_critic_batch(data,actor,critic,target_actor,target_critic,params)
            losses.append(critic_loss)
    mongo.close()
    return actor,critic,params

def train_batch(rank,env_params,training_params,learning_params,network_params,validation_params):
    world_size = 2
    if torch.cuda.device_count() > 1:
        setup_world(rank,world_size)
    env = Poker(env_params)
    # Setup for dual gpu and mp parallel training
    actor,critic,target_actor,target_critic = instantiate_models(rank,training_params,learning_params,network_params)
    for e in range(training_params['training_epochs']):
        if validation_params['koth']:
            generate_vs_frozen(env,target_actor,target_critic,villain,training_params,rank)
        else:
            generate_trajectories(env,target_actor,target_critic,training_params,rank)
        actor,critic,learning_params = batch_learning_update(rank,actor,critic,target_actor,target_critic,learning_params)
        training_params['training_round'] += 1
        learning_params['training_round'] += 1
        # if e % training_params['save_every'] == 0 and rank == 0:
        #     torch.save(actor.state_dict(), os.path.join(training_params['actor_path'],f'OmahaActor_{e}'))
        #     torch.save(critic.state_dict(), os.path.join(training_params['critic_path'],f'OmahaCritic_{e}'))
    if torch.cuda.device_count() > 1:
        dist.barrier()
        cleanup()

def train_combined(env,model,training_params,learning_params,rank):
    for e in range(training_params['training_epochs']):
        sys.stdout.write('\r')
        generate_trajectories(env,model,training_params,rank)
        # train on trajectories
        model,learning_params = combined_learning_update(model,learning_params)
        sys.stdout.write("[%-60s] %d%%" % ('='*(60*(e+1)//training_params['training_epochs']), (100*(e+1)//training_params['training_epochs'])))
        sys.stdout.flush()
        sys.stdout.write(f", epoch {(e+1):.2f}, Training round {training_params['training_round']}, ID: {rank}")
        sys.stdout.flush()
        training_params['training_round'] += 1
        learning_params['training_round'] += 1
        if e % training_params['save_every'] == 0 and rank == 0:
            torch.save(model.state_dict(), os.path.join(training_params['save_dir'],f'OmahaCombined_{e}'))

def instantiate_models(rank,training_params,learning_params,network_params):
    network_params['device'] = rank
    learning_params['device'] = rank
    seed = network_params['seed']
    nS = network_params['nS']
    nA = network_params['nA']
    nB = network_params['nB']
    actor = OmahaActor(seed,nS,nA,nB,network_params).to(rank)
    critic = OmahaObsQCritic(seed,nS,nA,nB,network_params).to(rank)
    ddp_actor = DDP(actor,device_ids=[rank],find_unused_parameters=True)
    ddp_critic = DDP(critic,device_ids=[rank],find_unused_parameters=True)
    latest_actor_path = return_latest_training_model_path(training_params['actor_path'])
    latest_critic_path = return_latest_training_model_path(training_params['critic_path'])
    load_weights(ddp_actor,latest_actor_path,rank,ddp=True)
    load_weights(ddp_critic,latest_critic_path,rank,ddp=True)
    # target networks
    target_actor = OmahaActor(seed,nS,nA,nB,network_params).to(rank)
    target_critic = OmahaObsQCritic(seed,nS,nA,nB,network_params).to(rank)
    ddp_target_actor = DDP(target_actor,device_ids=[rank],find_unused_parameters=True)
    ddp_target_critic = DDP(target_critic,device_ids=[rank],find_unused_parameters=True)
    hard_update(ddp_actor,ddp_target_actor)
    hard_update(ddp_critic,ddp_target_critic)
    actor_optimizer = optim.Adam(ddp_actor.parameters(), lr=network_params['actor_lr'],weight_decay=network_params['L2'])
    critic_optimizer = optim.Adam(ddp_critic.parameters(), lr=network_params['critic_lr'])
    learning_params['actor_optimizer'] = actor_optimizer
    learning_params['critic_optimizer'] = critic_optimizer
    return ddp_actor,ddp_critic,ddp_target_actor,ddp_target_critic

def train_dual(rank,env_params,training_params,learning_params,network_params,validation_params):
    if torch.cuda.device_count() > 1:
        world_size = 2
        setup_world(rank,world_size)
    env = Poker(env_params)
    # Setup for dual gpu and mp parallel training
    actor,critic,target_actor,target_critic = instantiate_models(rank,training_params,learning_params,network_params)
    if validation_params['koth']:
        villain = load_villain(rank,network_params,training_params['baseline_path'])
    for e in range(training_params['training_epochs']):
        if validation_params['koth']:
            generate_vs_frozen(env,target_actor,target_critic,villain,training_params,rank)
        else:
            generate_trajectories(env,target_actor,target_critic,training_params,rank)
        # train on trajectories
        dist.barrier()
        dual_learning_update(rank,actor,critic,target_actor,target_critic,learning_params,validation_params)
        training_params['training_round'] += 1
        learning_params['training_round'] += 1
        if (e+1) % training_params['save_every'] == 0 and rank == 0:
            torch.save(actor.state_dict(), os.path.join(training_params['actor_path'],f'OmahaActor_{e}'))
            torch.save(critic.state_dict(), os.path.join(training_params['critic_path'],f'OmahaCritic_{e}'))
    if rank == 0:
        torch.save(actor.state_dict(), os.path.join(training_params['actor_path'],'OmahaActorFinal'))
        torch.save(critic.state_dict(), os.path.join(training_params['critic_path'],'OmahaCriticFinal'))
        print(f"Saved model weights to {os.path.join(training_params['actor_path'],'OmahaActorFinal')} and {os.path.join(training_params['critic_path'],'OmahaCriticFinal')}")
    if torch.cuda.device_count() > 1:
        dist.barrier()
        cleanup()