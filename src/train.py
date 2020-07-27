import os
import poker.datatypes as pdt
import models.network_config as ng
import copy
import torch
import torch.nn.functional as F
import sys
import numpy as np
from pymongo import MongoClient
from collections import defaultdict
from hand_recognition.data_loader import datasetLoader
import copy
import time

from db import MongoDB
from poker.env import Poker
from agents.agent import ParallelAgent,FullAgent

def pad_state(state,maxlen):
    N = maxlen - state.shape[1]
    padding = np.zeros(N)
    return padded_state

def generate_trajectories(env,actor,training_params,id):
    """We want to store """
    actor = actor.eval()
    trajectories = defaultdict(lambda:[])
    for e in range(training_params['epochs']):
        trajectory = defaultdict(lambda:{'states':[],'obs':[],'betsize_masks':[],'action_masks':[], 'actions':[],'action_category':[],'action_probs':[],'action_prob':[],'betsize':[],'rewards':[]})
        state,obs,done,action_mask,betsize_mask = env.reset()
        cur_player = env.current_player
        trajectory[cur_player]['states'].append(copy.copy(state))
        trajectory[cur_player]['action_masks'].append(copy.copy(action_mask))
        trajectory[cur_player]['betsize_masks'].append(copy.copy(betsize_mask))
        while not done:
            actor_outputs = actor(state,action_mask,betsize_mask)
            trajectory[cur_player]['actions'].append(actor_outputs['action'])
            trajectory[cur_player]['action_category'].append(actor_outputs['action_category'])
            trajectory[cur_player]['action_prob'].append(actor_outputs['action_prob'])
            trajectory[cur_player]['action_probs'].append(actor_outputs['action_probs'])
            trajectory[cur_player]['betsize'].append(actor_outputs['betsize'])
            state,obs,done,action_mask,betsize_mask = env.step(actor_outputs)
            cur_player = env.current_player
            if not done:
                trajectory[cur_player]['states'].append(state)
                trajectory[cur_player]['action_masks'].append(action_mask)
                trajectory[cur_player]['betsize_masks'].append(betsize_mask)
        assert len(trajectory[cur_player]['betsize']) == len(trajectory[cur_player]['betsize_masks'])
        rewards = env.player_rewards()
        for position in trajectory.keys():
            N = len(trajectory[position]['betsize_masks'])
            trajectory[position]['rewards'] = [rewards[position]] * N
            trajectories[position].append(trajectory[position])
    insert_tic = time.time()
    insert_data(trajectories,env.state_mapping,env.obs_mapping,training_params['training_round'],training_params['game'],id,training_params['epochs'])
    insert_toc = time.time()
    print(f'insert data {insert_toc - insert_tic}')

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
                    'action_probs':action_probs[step].tolist(),
                    'action_prob':action_prob[step].tolist(),
                    'action':actions[step],
                    'action_category':action_categories[step],
                    'betsize_mask':betsize_masks[step].tolist(),
                    'action_mask':action_masks[step].tolist(),
                    'betsize':betsizes[step],
                    'reward':rewards[step]
                }
                db['game_data'].insert_one(state_json)
    client.close()

def return_value_mask(actions):
    # print('actions',actions)
    M = 1#actions.shape[0]
    value_mask = torch.zeros(M,5)
    value_mask[torch.arange(M),actions] = 1
    value_mask = value_mask.bool()
    return value_mask.squeeze(0)

def scale_rewards(self,rewards,factor=1):
    """Scales rewards between -1 and 1, with optional factor to increase valuation differences"""
    return (2 * ((rewards + self.min_reward) / (self.max_reward + self.min_reward)) - 1) * factor

def learning_update(actor,critic,params):
    actor = actor.train()
    device = params['device']
    critic_optimizer = params['critic_optimizer']
    actor_optimizer = params['actor_optimizer']
    query = {'training_round':params['training_round']}
    projection = {'state':1,'betsize_mask':1,'action_mask':1,'action':1,'reward':1,'_id':0}
    db_tic = time.time()
    client = MongoClient('localhost', 27017,maxPoolSize=10000)
    db = client['poker']
    data = db['game_data'].find(query,projection)
    db_toc = time.time()
    print(f'db retrieval {db_toc - db_tic}')
    # losses = []
    for poker_round in data:
        loop_tic = time.time()
        torch_tic = time.time()
        state = torch.tensor(poker_round['state'],dtype=torch.float32).to(device)
        action = torch.tensor(poker_round['action'],dtype=torch.long).to(device)
        reward = torch.tensor(poker_round['reward'],dtype=torch.float32).to(device)
        betsize_mask = torch.tensor(poker_round['betsize_mask'],dtype=torch.long).to(device)
        action_mask = torch.tensor(poker_round['action_mask'],dtype=torch.long).to(device)
        torch_toc = time.time()
        print(f'torch conversion {torch_toc - torch_tic}')
        ## Critic update ##
        critic_tic = time.time()
        critic_forward_tic = time.time()
        local_values = critic(state)['value']
        critic_forward_toc = time.time()
        print(f'critic forward {critic_forward_toc - critic_forward_tic}')
        value_mask = return_value_mask(action)
        TD_error = local_values[value_mask] - reward
        critic_loss = (TD_error**2*0.5).mean()
        # critic_loss = F.smooth_l1_loss(reward,TD_error,reduction='sum')
        critic_optimizer.zero_grad()
        critic_backward_tic = time.time()
        critic_loss.backward()
        critic_backward_toc = time.time()
        print(f'critic backward {critic_backward_toc - critic_backward_tic}')
        critic_grad_tic = time.time()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), params['gradient_clip'])
        critic_grad_toc = time.time()
        print(f'critic grad {critic_grad_toc - critic_grad_tic}')
        critic_optimizer_tic = time.time()
        critic_optimizer.step()
        critic_optimizer_toc = time.time()
        print(f'critic optimizer {critic_optimizer_toc - critic_optimizer_tic}')
        critic_toc = time.time()
        print(f'critic update {critic_toc - critic_tic}')
        # losses.append(critic_loss.item())
        # print('local_values',local_values[value_mask],reward)
        # Agent.soft_update(local_critic,target_critic,tau)

        # Actor update #
        actor_tic = time.time()
        target_values = critic(state)['value']
        actor_out = actor(state,action_mask,betsize_mask)
        expected_value = (actor_out['action_probs'].view(-1) * target_values.view(-1)).view(value_mask.size()).detach().sum(-1)
        advantages = (target_values[value_mask] - expected_value).view(-1)
        policy_loss = (-actor_out['action_prob'].view(-1) * advantages).sum()
        actor_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), params['gradient_clip'])
        actor_optimizer.step()
        loop_toc = time.time()
        actor_toc = time.time()
        print(f'actor update {actor_toc - actor_tic}')
        print(f'learning loop {loop_toc - loop_tic}')
        # Agent.soft_update(self.actor,self.target_actor,self.tau)
        # loss_dict[i] = sum(losses)
    # print('loss sum',sum(losses))
    return actor,critic,params

def train(env,actor,critic,training_params,learning_params,id):
    for e in range(training_params['training_epochs']):
        learning_params['training_round'] = e
        training_params['training_round'] = e
        trajectory_tic = time.time()
        generate_trajectories(env,actor,training_params,id)
        trajectory_toc = time.time()
        print(f'trajectory time {trajectory_toc - trajectory_tic}')
        # train on trajectories
        learning_tic = time.time()
        actor,critic,learning_params = learning_update(actor,critic,learning_params)
        learning_toc = time.time()
        print(f'learning time {learning_toc - learning_tic}')
        print(f'Training round {e}, ID {id}')