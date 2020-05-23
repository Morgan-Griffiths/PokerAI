import os
import kuhn.datatypes as pdt
import models.network_config as ng
import copy
import torch
import sys

from db import MongoDB
from models.buffers import PrioritizedReplayBuffer
from kuhn.env import Poker
from agents.agent import Agent,Priority_DQN,return_agent,ParallelAgent

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
    env = Poker(env_params)
    nS = env.state_space
    nO = env.observation_space
    nA = env.action_space
    nB = env.betsize_space
    nC = nA - 2 + nB
    print_every = (training_params['epochs']+1) // 5
    seed = 154
    agent = ParallelAgent(nS,nO,nA,nB,seed,agent_params,actor,critic)
    training_data = copy.deepcopy(training_params['training_data'])
    for e in range(1,training_params['epochs']+1):
        last_state,state,obs,done,mask,betsize_mask = env.reset()
        while not done:
            if env.game == pdt.GameTypes.HISTORICALKUHN:
                actor_outputs = agent(state,mask,betsize_mask) if env.rules.betsize == True else agent(state,mask)
            else:
                actor_outputs = agent(last_state,mask,betsize_mask) if env.rules.betsize == True else agent(last_state,mask)
            last_state,state,obs,done,mask,betsize_mask = env.step(actor_outputs)
        ml_inputs = env.ml_inputs()
        agent.learn(ml_inputs)
        ml_inputs = detach_ml(ml_inputs)
        for position in ml_inputs.keys():
            training_data[position].append(ml_inputs[position])
        if e % print_every == 0:
            print(f'PID {pid}, Epoch {e}')
    mongo = MongoDB()
    mongo.clean_db()
    mongo.store_data(training_data,env.db_mapping,training_params['training_round'],env.game)

def gather_trajectories(agent_params,env_params,training_params,id,return_dict):
    pid = os.getpid()
    print(f"Intiantiating process ID {id}, pid {pid}")
    env = Poker(env_params)
    nS = env.state_space
    nO = env.observation_space
    nA = env.action_space
    nB = env.betsize_space
    nC = nA - 2 + nB
    seed = 154
    agent = return_agent(training_params['agent_type'],nS,nO,nA,nB,seed,agent_params)
    training_data = copy.deepcopy(training_params['training_data'])
    for e in range(1,training_params['epochs']+1):
        state,obs,done,mask,betsize_mask = env.reset()
        while not done:
            actor_outputs = agent(state,mask,betsize_mask) if env.rules.betsize == True else agent(state,mask)
            state,obs,done,mask,betsize_mask = env.step(actor_outputs)
        ml_inputs = env.ml_inputs(serialize=True)
        agent.learn(ml_inputs)
        for position in ml_inputs.keys():
            training_data[position].append(ml_inputs[position])
        if e % 100 == 0:
            print(f'ID {id}, PID {pid} Epoch {e}') 
    return_dict[id] = training_data