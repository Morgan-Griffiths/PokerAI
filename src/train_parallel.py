import os
import poker.datatypes as pdt
import models.network_config as ng
import copy
import torch
import sys
from pymongo import MongoClient

from db import MongoDB
from models.buffers import PrioritizedReplayBuffer
from poker.multistreet_env import MSPoker
from agents.agent import ParallelAgent,FullAgent

def insert_data(training_data:dict,mapping:dict,training_round:int,gametype:str,id:int,epochs:int):
    """
    training_data, contains all positions
    Poker db;
    state,obs,action,log_prob,reward collections
    State:
    training_run,round,step,p1_hand,previous_action
    *** future ***
    p1_position,p1_hand,p1_stack
    p2_position,p2_stack
    pot,board_cards,previous_action
    Action:
    training_run,round,step,p1_action,log_prob
    Reward:
    training_run,round,step,reward
    """
    client = MongoClient('localhost', 27017,maxPoolSize=10000)
    db = client['poker']
    keys = training_data.keys()
    positions = [position for position in keys if position in ['SB','BB'] ]   
    for position in positions:
        for i,poker_round in enumerate(training_data[position]):
            game_states = poker_round['game_states']
            observations = poker_round['observations']
            actions = poker_round['actions']
            action_prob = poker_round['action_prob']
            action_probs = poker_round['action_probs']
            rewards = poker_round['rewards']
            values = poker_round['values']
            betsizes = poker_round['betsizes']
            betsize_prob = poker_round['betsize_prob']
            betsize_probs = poker_round['betsize_probs']
            hand_strength = poker_round['hand_strength']
            assert(isinstance(rewards,torch.Tensor))
            assert(isinstance(actions,torch.Tensor))
            assert(isinstance(action_prob,torch.Tensor))
            assert(isinstance(action_probs,torch.Tensor))
            assert(isinstance(observations,torch.Tensor))
            assert(isinstance(game_states,torch.Tensor))
            for step,game_state in enumerate(game_states):
                board = game_state[-1,mapping['state']['board']].view(-1)
                hand = game_state[0,mapping['state']['hand']].view(-1)
                vil_hand = game_state[0,mapping['observation']['vil_hand']].view(-1)
                previous_action = game_state[:,mapping['state']['previous_action']].view(-1)
                street = game_state[:,mapping['state']['street']].view(-1)
                state_json = {
                    'position':position,
                    'street':street.tolist(),
                    'hand':hand.tolist(),
                    'vil_hand':vil_hand.tolist(),
                    'reward':rewards[step].tolist(),
                    'action':actions[step].tolist(),
                    'action_probs':action_probs[step].detach().tolist(),
                    'previous_action':previous_action.tolist(),
                    'training_round':training_round,
                    'poker_round':i + (id * epochs),
                    'step':step,
                    'game':gametype,
                    'hand_strength':hand_strength
                    }
                if len(betsizes) > 0:
                    if betsizes[step][0].dim() > 1:
                        index = torch.arange(betsizes[step].size(0))
                        state_json['betsizes'] = float(betsizes[step][index,actions[step]].detach())
                        if len(betsize_prob) > 0:
                            state_json['betsize_prob'] = float(betsize_prob[step][index,actions[step]].detach())
                            state_json['betsize_probs'] = betsize_probs[step][index,actions[step]].detach().tolist()
                    else:
                        state_json['betsizes'] = betsizes[step].detach().tolist()
                        if len(betsize_prob) > 0:
                            state_json['betsize_prob'] = float(betsize_prob[step].detach())
                            state_json['betsize_probs'] = betsize_probs[step].detach().tolist()
                if len(values) > 0:
                    if len(values[step][0]) > 1:
                        index = torch.arange(values[step].size(0))
                        state_json['value'] = values[step][index,actions[step]].detach().tolist()
                    else:
                        state_json['value'] = float(values[step].detach())
                db['game_data'].insert_one(state_json)
    client.close()

def detach_ml(ml_inputs):
    for position in ml_inputs.keys():
        ml_inputs[position]['action_probs'] = ml_inputs[position]['action_probs'].detach()
        ml_inputs[position]['action_prob'] = ml_inputs[position]['action_prob'].detach()
    return ml_inputs

def train_shared_model(agent_params,env_params,training_params,id,actor,critic):
    actor.train()
    critic.train()
    pid = os.getpid()
    print(f"Intiantiating process PID {pid}")
    env = MSPoker(env_params)
    nS = env.state_space
    nO = env.observation_space
    nA = env.action_space
    nB = env.betsize_space
    nC = nA - 2 + nB
    print_every = (training_params['epochs']+1) // 5
    seed = 154
    agent = FullAgent(nS,nO,nA,nB,seed,agent_params,actor,critic)
    training_data = copy.deepcopy(training_params['training_data'])
    for e in range(1,training_params['epochs']+1):
        state,obs,done,mask,betsize_mask = env.reset()
        while not done:
            actor_outputs = agent(state,mask,betsize_mask)
            state,obs,done,mask,betsize_mask = env.step(actor_outputs)
        ml_inputs = env.ml_inputs()
        agent.learn(ml_inputs)
        ml_inputs = detach_ml(ml_inputs)
        for position in ml_inputs.keys():
            training_data[position].append(ml_inputs[position])
        if id == 0 and e % print_every == 0:
            print(f'PID {pid}, Epoch {e}')
    insert_data(training_data,env.db_mapping,training_params['training_round'],env.game,id,training_params['epochs'])