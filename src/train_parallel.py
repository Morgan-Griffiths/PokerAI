import os
import poker.datatypes as pdt
import models.network_config as ng
import copy
import torch
import sys

from models.buffers import PrioritizedReplayBuffer
from kuhn.env import Poker
from agents.agent import Agent,Priority_DQN,return_agent

def gather_trajectories(agent_params,env_params,training_params,id,return_dict):
    print(f"Intiantiating process id {id}")
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
        for position in ml_inputs.keys():
            training_data[position].append(ml_inputs[position])
        if e % 100 == 0:
            print(f'ID {id}, Epoch {e}') 
    return_dict[id] = training_data