
from torch import optim
import torch.nn.functional as F
import torch
import os
from pymongo import MongoClient
import numpy as np
import sys

from db import MongoDB
from models.model_utils import scale_rewards
from poker_env.config import Config
import poker_env.datatypes as pdt
from poker_env.env import Poker
from train import generate_trajectories,dual_learning_update,combined_learning_update
from models.networks import CombinedNet,OmahaActor,OmahaQCritic

def return_value_mask(actions):
    M = 1#actions.shape[0]
    value_mask = torch.zeros(M,5)
    value_mask[torch.arange(M),actions] = 1
    value_mask = value_mask.bool()
    return value_mask.squeeze(0)

def eval_critic(critic,params):
    critic_optimizer = params['critic_optimizer']  
    device = params['device']
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
            state = poker_round['state']
            action = poker_round['action']
            reward = torch.tensor(poker_round['reward']).unsqueeze(-1)
            betsize_mask = poker_round['betsize_mask']
            action_mask = poker_round['action_mask']
            ## Critic update ##
            local_values = critic(state)['value']
            value_mask = return_value_mask(action)
            # print('local_values',local_values)
            # print('value_mask,action',value_mask,action)
            # print('local_values[value_mask],reward',local_values[value_mask],reward)
            TD_error = local_values[value_mask] - reward
            # critic_loss = (TD_error**2*0.5).mean()
            critic_loss = F.smooth_l1_loss(reward,TD_error,reduction='sum')
            # print('critic_loss',critic_loss)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()
            losses.append(critic_loss.item())   
            # Agent.soft_update(local_critic,target_critic,tau)
            sys.stdout.write("[%-60s] %d%%" % ('='*(60*(j)//len(data)), (100*(j)//len(data))))
            sys.stdout.flush()
            sys.stdout.write(f", round {(j):.2f}")
            sys.stdout.flush()
            # Agent.soft_update(self.actor,self.target_actor,self.tau)
        print(f'Training Round {i}, critic loss {sum(losses)}')
    del data

def eval_network_updates(actor,critic,params):
    critic_optimizer = params['critic_optimizer']  
    actor_optimizer = params['actor_optimizer']  
    device = params['device']
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
            state = poker_round['state']
            action = poker_round['action']
            reward = torch.tensor(poker_round['reward']).unsqueeze(-1)
            betsize_mask = poker_round['betsize_mask']
            action_mask = poker_round['action_mask']
            ## Critic update ##
            local_values = critic(state)['value']
            value_mask = return_value_mask(action)
            # print('local_values',local_values)
            # print('value_mask,action',value_mask,action)
            # print('local_values[value_mask],reward',local_values[value_mask],reward)
            TD_error = local_values[value_mask] - reward
            # critic_loss = (TD_error**2*0.5).mean()
            critic_loss = F.smooth_l1_loss(reward,TD_error,reduction='sum')
            # print('critic_loss',critic_loss)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()
            losses.append(critic_loss.item())   
            # Agent.soft_update(local_critic,target_critic,tau)
            # Actor update #
            target_values = critic(state)['value']
            actor_out = actor(np.array(state),np.array(action_mask),np.array(betsize_mask))
            actor_value_mask = return_value_mask(actor_out['action'])
            expected_value = (actor_out['action_probs'].view(-1) * target_values.view(-1)).view(actor_value_mask.size()).detach().sum(-1)
            advantages = (target_values[actor_value_mask] - expected_value).view(-1)
            policy_loss = (-actor_out['action_prob'].view(-1) * advantages).sum()
            actor_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), params['gradient_clip'])
            actor_optimizer.step()
            policy_losses.append(policy_loss)
            sys.stdout.write("[%-60s] %d%%" % ('='*(60*(j)//len(data)), (100*(j)//len(data))))
            sys.stdout.flush()
            sys.stdout.write(f", round {(j):.2f}")
            sys.stdout.flush()
            # Agent.soft_update(self.actor,self.target_actor,self.tau)
        # print('\nprobs,prob,actor action,original action',actor_out['action_probs'].detach(),actor_out['action_prob'].detach(),actor_out['action'],action)
        # print('\nlocal_values,Q_value',local_values,local_values[value_mask].item())
        # print('\ntarget_values,target_Q_value',target_values,target_values[value_mask].item())
        # print('\ntarget_values*mask',(actor_out['action_probs'].view(-1) * target_values.view(-1)).view(value_mask.size()))
        # print('\nexpected_value',expected_value)
        # print('\nadvantages',advantages)
        # print('\nreward',reward)
        # print('\npolicy_loss',policy_loss)
        print(f'\nTraining Round {i}, critic loss {sum(losses)}, policy loss {sum(policy_losses)}')
    del data

def eval_combined_updates(model,params):
    optimizer = params['model_optimizer']  
    device = params['device']
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
            state = poker_round['state']
            action = poker_round['action']
            reward = torch.tensor(poker_round['reward']).unsqueeze(-1)
            betsize_mask = poker_round['betsize_mask']
            action_mask = poker_round['action_mask']
            ## Critic update ##
            local_values = critic(state)['value']
            value_mask = return_value_mask(action)
            # print('local_values',local_values)
            # print('value_mask,action',value_mask,action)
            # print('local_values[value_mask],reward',local_values[value_mask],reward)
            TD_error = local_values[value_mask] - reward
            # critic_loss = (TD_error**2*0.5).mean()
            critic_loss = F.smooth_l1_loss(reward,TD_error,reduction='sum')
            # print('critic_loss',critic_loss)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()
            losses.append(critic_loss.item())   
            # Agent.soft_update(local_critic,target_critic,tau)
            # Actor update #
            target_values = critic(state)['value']
            actor_out = actor(np.array(state),np.array(action_mask),np.array(betsize_mask))
            expected_value = (actor_out['action_probs'].view(-1) * target_values.view(-1)).view(value_mask.size()).detach().sum(-1)
            advantages = (target_values[value_mask] - expected_value).view(-1)
            policy_loss = (-actor_out['action_prob'].view(-1) * advantages).sum()
            actor_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), params['gradient_clip'])
            actor_optimizer.step()
            policy_losses.append(policy_loss)
            sys.stdout.write("[%-60s] %d%%" % ('='*(60*(j)//len(data)), (100*(j)//len(data))))
            sys.stdout.flush()
            sys.stdout.write(f", round {(j):.2f}")
            sys.stdout.flush()
            # Agent.soft_update(self.actor,self.target_actor,self.tau)
        print(f'Training Round {i}, critic loss {sum(losses)}, policy loss {sum(policy_losses)}')
    del data

if __name__ == "__main__":
    import argparse

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
        'obs_mapping':config.obs_mapping,
        'embedding_size':128,
        'transformer_in':1280,
        'transformer_out':128,
        'device':device,
        'frozen_layer_path':'../hand_recognition/checkpoints/regression/PartialHandRegression'
    }
    training_params = {
        'training_epochs':50,
        'epochs':10,
        'training_round':0,
        'game':'Omaha',
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


    # Instantiate network
    # alphaPoker = CombinedNet(seed,nS,nA,nB,network_params).to(device)
    # alphaPoker_optimizer = optim.Adam(alphaPoker.parameters(), lr=config.agent_params['critic_lr'])
    # learning_params['model_optimizer'] = alphaPoker_optimizer
    actor = OmahaActor(seed,nS,nA,nB,network_params).to(device)
    critic = OmahaQCritic(seed,nS,nA,nB,network_params).to(device)
    actor_optimizer = optim.Adam(actor.parameters(), lr=config.agent_params['actor_lr'],weight_decay=config.agent_params['L2'])
    critic_optimizer = optim.Adam(critic.parameters(), lr=3e-5)
    learning_params['actor_optimizer'] = actor_optimizer
    learning_params['critic_optimizer'] = critic_optimizer

    # Clean mongo
    # mongo = MongoDB()
    # mongo.clean_db()
    # mongo.close()
    # # Gen trajectories
    # generate_trajectories(env,actor,training_params,id=0)

    # Eval learning models
    eval_network_updates(actor,critic,learning_params)
    # eval_critic(critic,learning_params)