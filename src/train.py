import os
import poker.datatypes as pdt
import models.network_config as ng
import copy
import torch
import sys
# import tdqm

def train(env,agent,training_params):
    training_data = copy.deepcopy(training_params['training_data'])
    for e in range(training_params['epochs']):
        sys.stdout.write('\r')
        state,obs,done = env.reset()
        while not done:
            mask,betsize_mask = env.action_mask(state)
            env.players.store_masks(mask,betsize_mask)
            actor_outputs = agent(state,mask,betsize_mask) if env.rules.betsize == True else agent(state,mask)
            if training_params['agent_type'] == pdt.AgentTypes.ACTOR_CRITIC:
                critic_outputs = agent.critique(obs,actor_outputs['action'])
                env.players.store_values(critic_outputs)
            state,obs,done = env.step(actor_outputs)
        ml_inputs = env.ml_inputs()
        agent.learn(ml_inputs)
        for position in ml_inputs.keys():
            training_data[position].append(ml_inputs[position])
        
        sys.stdout.write("[%-60s] %d%%" % ('='*(60*(e+1)//training_params['epochs']), (100*(e+1)//training_params['epochs'])))
        sys.stdout.flush()
        sys.stdout.write(", epoch %d"% (e+1))
        sys.stdout.flush()
    agent.save_weights(os.path.join(training_params['save_dir'],training_params['agent_name']))
    return training_data