import os
import poker_env.datatypes as pdt
import models.network_config as ng
import copy
import torch
import torch.nn.functional as F
import sys
import numpy as np
from pymongo import MongoClient
from collections import defaultdict
import copy
import time
import logging

from models.networks import BetAgent
from models.model_updates import update_actor_critic,update_combined,update_critic_batch,update_actor_critic_batch
from utils.data_loaders import return_trajectoryloader
from models.model_utils import scale_rewards,soft_update
from tournament import tournament
from db import MongoDB
from poker_env.env import Poker
import torch.autograd.profiler as profiler

def pad_state(state,maxlen):
    N = maxlen - state.shape[1]
    padding = np.zeros(N)
    return padded_state

def generate_vs_frozen(env,actor,critic,villain,training_params,id):
    # actor.eval()
    # critic.eval()
    trajectories = defaultdict(lambda:[])
    with torch.no_grad():
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
                actor_outputs = actor_positions[env.current_player](state,action_mask,betsize_mask)
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
    insert_data(trajectories,env.state_mapping,env.obs_mapping,training_params['training_round'],training_params['game'],id,training_params['generate_epochs'])


def generate_trajectories(env,actor,critic,training_params,id):
    """We want to store """
    # actor.eval()
    # critic.eval()
    with torch.no_grad():
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
                actor_outputs = actor(state,action_mask,betsize_mask)
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
    insert_data(trajectories,env.state_mapping,env.obs_mapping,training_params['training_round'],training_params['game'],id,training_params['generate_epochs'])

def insert_data(training_data:dict,mapping:dict,obs_mapping,training_round:int,gametype:str,id:int,epochs:int):
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
                    'training_round':training_round,
                    'poker_round':i + (id * epochs),
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

def dual_learning_update(actor,critic,target_actor,target_critic,params):
    mongo = MongoDB()
    actor.train()
    query = {'training_round':params['training_round']}
    projection = {'obs':1,'state':1,'betsize_mask':1,'action_mask':1,'action':1,'reward':1,'_id':0}
    data = list(mongo.get_data(query,projection))
    # loss_dict = defaultdict(lambda:None)
    for i in range(params['learning_rounds']):
        policy_losses = []
        losses = []
        for poker_round in data:
            critic_loss,policy_loss = update_actor_critic(poker_round,critic,target_critic,actor,target_actor,params)
            losses.append(critic_loss)
            policy_losses.append(policy_loss)
        # print(f'Learning Round {i}, critic loss {sum(losses)}, policy loss {sum(policy_losses)}')
    mongo.close()
    del data
    return actor,critic,params

def batch_learning_update(actor,critic,target_actor,target_critic,params):
    mongo = MongoDB()
    actor.train()
    query = {'training_round':params['training_round']}
    projection = {'obs':1,'state':1,'betsize_mask':1,'action_mask':1,'action':1,'reward':1,'_id':0}
    db_data = mongo.get_data(query,projection)
    trainloader = return_trajectoryloader(db_data)
    for _ in range(params['learning_rounds']):
        losses = []
        for i,data in enumerate(trainloader):
            critic_loss = update_actor_critic_batch(data,actor,critic,target_actor,target_critic,params)
            losses.append(critic_loss)
        # print(f'Learning Round {i}, critic loss {sum(losses)}, policy loss {sum(policy_losses)}')
    mongo.close()
    return actor,critic,params

def train_batch(env,actor,critic,target_actor,target_critic,training_params,learning_params,id):
    villain = BetAgent()
    for e in range(training_params['training_epochs']):
        sys.stdout.write('\r')
        generate_vs_frozen(env,target_actor,target_critic,villain,training_params,id)
        # generate_trajectories(env,target_actor,target_critic,training_params,id)
        # train on trajectories
        actor,critic,learning_params = batch_learning_update(actor,critic,target_actor,target_critic,learning_params)
        sys.stdout.write("[%-60s] %d%%" % ('='*(60*(e+1)//training_params['training_epochs']), (100*(e+1)//training_params['training_epochs'])))
        sys.stdout.flush()
        sys.stdout.write(f", epoch {(e+1):.2f}, Training round {training_params['training_round']}, ID: {id}")
        sys.stdout.flush()
        training_params['training_round'] += 1
        learning_params['training_round'] += 1
        learning_params['actor_lrscheduler'].step()
        learning_params['critic_lrscheduler'].step()
        if e % training_params['save_every'] == 0 and id == 0:
            torch.save(actor.state_dict(), os.path.join(training_params['actor_path'],f'OmahaActor_{e}'))
            torch.save(critic.state_dict(), os.path.join(training_params['critic_path'],f'OmahaCritic_{e}'))

def train_combined(env,model,training_params,learning_params,id):
    for e in range(training_params['training_epochs']):
        sys.stdout.write('\r')
        generate_trajectories(env,model,training_params,id)
        # train on trajectories
        model,learning_params = combined_learning_update(model,learning_params)
        sys.stdout.write("[%-60s] %d%%" % ('='*(60*(e+1)//training_params['training_epochs']), (100*(e+1)//training_params['training_epochs'])))
        sys.stdout.flush()
        sys.stdout.write(f", epoch {(e+1):.2f}, Training round {training_params['training_round']}, ID: {id}")
        sys.stdout.flush()
        training_params['training_round'] += 1
        learning_params['training_round'] += 1
        if e % training_params['save_every'] == 0 and id == 0:
            torch.save(model.state_dict(), os.path.join(training_params['save_dir'],f'OmahaCombined_{e}'))

def train_dual(env,actor,critic,target_actor,target_critic,training_params,learning_params,id):
    villain = BetAgent()
    for e in range(training_params['training_epochs']):
        sys.stdout.write('\r')
        generate_vs_frozen(env,target_actor,target_critic,villain,training_params,id)
        # generate_trajectories(env,target_actor,target_critic,training_params,id)
        # train on trajectories
        actor,critic,learning_params = dual_learning_update(actor,critic,target_actor,target_critic,learning_params)
        sys.stdout.write("[%-60s] %d%%" % ('='*(60*(e+1)//training_params['training_epochs']), (100*(e+1)//training_params['training_epochs'])))
        sys.stdout.flush()
        sys.stdout.write(f", epoch {(e+1):.2f}, Training round {training_params['training_round']}, ID: {id}")
        sys.stdout.flush()
        training_params['training_round'] += 1
        learning_params['training_round'] += 1
        if e % training_params['save_every'] == 0 and id == 0:
            torch.save(actor.state_dict(), os.path.join(training_params['actor_path'],f'OmahaActor_{e}'))
            torch.save(critic.state_dict(), os.path.join(training_params['critic_path'],f'OmahaCritic_{e}'))