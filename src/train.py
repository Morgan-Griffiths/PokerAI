import torch
import os

def train(env,agent,training_params):
    training_data = training_params['training_data']
    for e in range(training_params['epochs']):
        state,obs,done = env.reset()
        mask = env.action_mask(state)
        while not done:
            action,log_probs = agent(state,mask)
            state,obs,done = env.step(action,log_probs)
            mask = env.action_mask(state)
        ml_inputs = env.ml_inputs()
        agent.learn(ml_inputs)
        for position in ml_inputs.keys():
            training_data[position].append(ml_inputs[position])
    agent.save_weights(os.path.join(training_params['save_dir'],training_params['agent_name']))
    return training_data