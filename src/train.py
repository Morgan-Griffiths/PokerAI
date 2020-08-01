import os
import poker.datatypes as pdt
import models.network_config as ng
from models.model_utils import soft_update
import copy
import torch
import torch.nn.functional as F
import sys
import numpy as np
from pymongo import MongoClient
from collections import defaultdict
from hand_recognition.data_loader import return_trajectoryloader
import copy
from tournament import tournament
import time
import logging

from db import MongoDB
from poker.env import Poker
from agents.agent import BetAgent

logging.basicConfig(level=logging.INFO)

def find_strength(strength):
    # 7462-6185 High card
    # 6185-3325 Pair
    # 3325-2467 2Pair
    # 2467-1609 Trips
    # 1609-1599  Stright
    # 1599-322 Flush
    # 322-166  FH
    # 166-10 Quads
    # 10-0 Str8 flush
    if strength > 6185:
        return 8
    if strength > 3325:
        return 7
    if strength > 2467:
        return 6
    if strength > 1609:
        return 5
    if strength > 1599:
        return 4
    if strength > 322:
        return 3
    if strength > 166:
        return 2
    if strength > 10:
        return 1
    return 0

def pad_state(state,maxlen):
    N = maxlen - state.shape[1]
    padding = np.zeros(N)
    return padded_state

def generate_trajectories(env,actor,training_params,id):
    """We want to store """
    action_probs = {'SB':defaultdict(lambda:[]),'BB':defaultdict(lambda:[])}
    actor = actor.eval()
    trajectories = defaultdict(lambda:[])
    for e in range(training_params['epochs']):
        trajectory = defaultdict(lambda:{'states':[],'obs':[],'betsize_masks':[],'action_masks':[], 'actions':[],'action_category':[],'action_probs':[],'action_prob':[],'betsize':[],'rewards':[]})
        state,obs,done,action_mask,betsize_mask = env.reset()
        cur_player = env.current_player
        trajectory[cur_player]['states'].append(copy.copy(state))
        trajectory[cur_player]['action_masks'].append(copy.copy(action_mask))
        trajectory[cur_player]['betsize_masks'].append(copy.copy(betsize_mask))
        i = 0
        while not done:
            actor_outputs = actor(state,action_mask,betsize_mask)
            trajectory[cur_player]['actions'].append(actor_outputs['action'])
            trajectory[cur_player]['action_category'].append(actor_outputs['action_category'])
            trajectory[cur_player]['action_prob'].append(actor_outputs['action_prob'])
            trajectory[cur_player]['action_probs'].append(actor_outputs['action_probs'])
            trajectory[cur_player]['betsize'].append(actor_outputs['betsize'])
            if e < 2:
                action_probs[cur_player][i].append(trajectory[cur_player]['action_probs'])
            state,obs,done,action_mask,betsize_mask = env.step(actor_outputs)
            cur_player = env.current_player
            if not done:
                trajectory[cur_player]['states'].append(state)
                trajectory[cur_player]['action_masks'].append(action_mask)
                trajectory[cur_player]['betsize_masks'].append(betsize_mask)
            i += 1
        assert len(trajectory[cur_player]['betsize']) == len(trajectory[cur_player]['betsize_masks'])
        rewards = env.player_rewards()
        for position in trajectory.keys():
            N = len(trajectory[position]['betsize_masks'])
            trajectory[position]['rewards'] = [rewards[position]] * N
            trajectory[position]['handstrength'] = [env.players[position].handrank] * N
            trajectories[position].append(trajectory[position])
    insert_data(trajectories,env.state_mapping,env.obs_mapping,training_params['training_round'],training_params['game'],id,training_params['epochs'])
    # print('SB',action_probs['SB'])
    # print('SB',handrank_probs['SB'])
    # print('BB',handrank_probs['BB'])
    # print('BB',action_probs['BB'])

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
            handstrength = poker_round['handstrength']
            assert(isinstance(rewards,list))
            assert(isinstance(actions,list))
            assert(isinstance(action_prob,list))
            assert(isinstance(action_probs,list))
            assert(isinstance(observations,list))
            assert(isinstance(states,list))
            for step,state in enumerate(states):
                state_json = {
                    'game':gametype,
                    'position':position,
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
                    'reward':rewards[step],
                    'hand_strength':handstrength[step]
                }
                db['game_data'].insert_one(state_json)
    client.close()

def return_value_mask(actions):
    M = actions.shape[0]
    value_mask = torch.zeros(M,5)
    value_mask[torch.arange(M),actions] = 1
    value_mask = value_mask.bool()
    return value_mask.squeeze(0)

def scale_rewards(self,rewards,factor=1):
    """Scales rewards between -1 and 1, with optional factor to increase valuation differences"""
    return (2 * ((rewards + self.min_reward) / (self.max_reward + self.min_reward)) - 1) * factor

def learning_update(local_actor,target_actor,local_critic,target_critic,params):
    log = logging.getLogger(__name__)
    local_actor = local_actor.train()
    device = params['device']
    critic_optimizer = params['critic_optimizer']
    actor_optimizer = params['actor_optimizer']
    query = {'training_round':params['training_round']}
    projection = {'state':1,'betsize_mask':1,'action_mask':1,'action':1,'reward':1,'_id':0}
    client = MongoClient('localhost', 27017,maxPoolSize=10000)
    db = client['poker']
    data = db['game_data'].find(query,projection)
    trainloader = return_trajectoryloader(data)
    # for _ in range(params['learning_rounds']):
    for i,data in enumerate(trainloader,1):
        # get the inputs; data is a list of [inputs, targets]
        loop_tic = time.time()
        trajectory, target = data.values()
        state = trajectory['state'].to(device)
        action = trajectory['action'].to(device)
        reward = target['reward'].to(device)
        betsize_mask = trajectory['betsize_mask'].to(device)
        action_mask = trajectory['action_mask'].to(device)
        # state = torch.tensor(poker_round['state'],dtype=torch.float32).to(device)
        # action = torch.tensor(poker_round['action'],dtype=torch.long).to(device)
        # reward = torch.tensor(poker_round['reward'],dtype=torch.float32).to(device)
        # betsize_mask = torch.tensor(poker_round['betsize_mask'],dtype=torch.long).to(device)
        # action_mask = torch.tensor(poker_round['action_mask'],dtype=torch.long).to(device)
        ## Critic update ##
        critic_tic = time.time()
        critic_forward_tic = time.time()
        local_values = local_critic(state)['value']
        critic_forward_toc = time.time()
        log.debug(f'critic forward {critic_forward_toc - critic_forward_tic}')
        value_mask = return_value_mask(action)
        TD_error = local_values[value_mask] - reward
        critic_loss = (TD_error**2*0.5).mean()
        # critic_loss = F.smooth_l1_loss(reward,TD_error,reduction='sum')
        critic_optimizer.zero_grad()
        critic_backward_tic = time.time()
        critic_loss.backward()
        critic_backward_toc = time.time()
        log.debug(f'critic backward {critic_backward_toc - critic_backward_tic}')
        critic_grad_tic = time.time()
        torch.nn.utils.clip_grad_norm_(local_critic.parameters(), params['gradient_clip'])
        critic_grad_toc = time.time()
        log.debug(f'critic grad {critic_grad_toc - critic_grad_tic}')
        critic_optimizer_tic = time.time()
        critic_optimizer.step()
        critic_optimizer_toc = time.time()
        log.debug(f'critic optimizer {critic_optimizer_toc - critic_optimizer_tic}')
        soft_update(local_critic,target_critic,tau=1e-1)
        critic_toc = time.time()
        log.debug(f'critic update {critic_toc - critic_tic}')
        # losses.append(critic_loss.item())
        # log.debug('local_values',local_values[value_mask],reward)

        # Actor update #
        actor_tic = time.time()
        target_values = target_critic(state)['value']
        actor_out = local_actor(state,action_mask,betsize_mask)
        expected_value = (actor_out['action_probs'].view(-1) * target_values.view(-1)).view(value_mask.size()).detach().sum(-1)
        advantages = (target_values[value_mask] - expected_value).view(-1)
        policy_loss = (-actor_out['action_prob'].view(-1) * advantages).sum()
        actor_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(local_actor.parameters(), params['gradient_clip'])
        actor_optimizer.step()
        loop_toc = time.time()
        actor_toc = time.time()
        soft_update(local_actor,target_actor,tau=1e-1)
        log.debug(f'actor update {actor_toc - actor_tic}')
        log.debug(f'learning loop {loop_toc - loop_tic}')
        # Agent.soft_update(self.actor,self.target_actor,self.tau)
    return local_actor,target_actor,local_critic,target_critic,params

def train(env,local_actor,target_actor,local_critic,target_critic,training_params,learning_params,eval_params,id):
    log = logging.getLogger(__name__)
    for e in range(training_params['training_epochs']):
        learning_params['training_round'] = e
        training_params['training_round'] = e
        trajectory_tic = time.time()
        generate_trajectories(env,target_actor,training_params,id)
        trajectory_toc = time.time()
        log.debug(f'trajectory time {trajectory_toc - trajectory_tic}')
        # train on trajectories
        learning_tic = time.time()
        local_actor,target_actor,local_critic,target_critic,learning_params = learning_update(local_actor,target_actor,local_critic,target_critic,learning_params)
        learning_toc = time.time()
        log.debug(f'learning time {learning_toc - learning_tic}')
        log.info(f'Training round {e}, ID {id}')
        # if e % training_params['evaluation_every'] == 0:
        #     results = tournament(env,actor,eval_agent,eval_params)
        #     print(results)
        #     print(f"Actor: {results['agent1']['SB'] + results['agent1']['BB']}")
        #     print(f"Eval_agent: {results['agent2']['SB'] + results['agent2']['BB']}")