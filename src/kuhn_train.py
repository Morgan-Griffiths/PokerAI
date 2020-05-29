import os
import kuhn.datatypes as pdt
import models.network_config as ng
import copy
import torch
import sys

# import tdqm

def train(env,agent,training_params):
    training_data = copy.deepcopy(training_params['training_data'])
    for e in range(training_params['epochs']):
        sys.stdout.write('\r')
        last_state,state,obs,done,mask,betsize_mask = env.reset()
        while not done:
            if env.game == pdt.GameTypes.HISTORICALKUHN:
                actor_outputs = agent(state,mask,betsize_mask) if env.rules.betsize == True else agent(state,mask)
            else:
                actor_outputs = agent(last_state,mask,betsize_mask) if env.rules.betsize == True else agent(last_state,mask)
            last_state,state,obs,done,mask,betsize_mask = env.step(actor_outputs)
        ml_inputs = env.ml_inputs()
        agent.learn(ml_inputs)
        for position in ml_inputs.keys():
            training_data[position].append(ml_inputs[position])
        training_data['action_records'].append(env.action_records)
        sys.stdout.write("[%-60s] %d%%" % ('='*(60*(e+1)//training_params['epochs']), (100*(e+1)//training_params['epochs'])))
        sys.stdout.flush()
        sys.stdout.write(", epoch %d"% (e+1))
        sys.stdout.flush()
    agent.save_weights(os.path.join(training_params['save_dir'],training_params['agent_name']))
    return training_data