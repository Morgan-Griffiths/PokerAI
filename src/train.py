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

def pad_state(state):
    return padded_state

def generate_trajectories():
    for e in range(training_params['epochs']):
        trajectory = []
        state,obs,done,mask,betsize_mask = env.reset()
        while not done:
            actor_outputs = agent(state,mask,betsize_mask)
            state,obs,done,mask,betsize_mask = env.step(actor_outputs)

def learning_update():
    pass

def train(env,agent,training_params):
    for e in range(training_params['epochs']):
        sys.stdout.write('\r')
        state,obs,done,mask,betsize_mask = env.reset()
        while not done:
            actor_outputs = agent(state,mask,betsize_mask)
            state,obs,done,mask,betsize_mask = env.step(actor_outputs)
        ml_inputs = env.ml_inputs()
        agent.learn(ml_inputs)
        for position in ml_inputs.keys():
            training_data[position].append(ml_inputs[position])
        training_data['action_records'].append(env.action_records)
        sys.stdout.write("[%-60s] %d%%" % ('='*(60*(e+1)//training_params['epochs']), (100*(e+1)//training_params['epochs'])))
        sys.stdout.flush()
        sys.stdout.write(", epoch %d"% (e+1))
        sys.stdout.flush()
    print(training_params['save_dir'])
    print(training_params['agent_name'])
    agent.save_weights(os.path.join(training_params['save_dir'],training_params['agent_name']))
    return training_data