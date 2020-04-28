import torch
import os
import poker.datatypes as pdt

def train(env,agent,training_params):
    training_data = training_params['training_data']
    for e in range(training_params['epochs']):
        state,obs,done = env.reset()
        while not done:
            mask = env.action_mask(state)
            action,log_probs,complete_probs = agent(state,mask)
            if training_params['agent_type'] == pdt.AgentTypes.ACTOR_CRITIC:
                value = agent.critique(obs,action)
                env.players.store_values(value)
            state,obs,done = env.step(action,log_probs,complete_probs)
        ml_inputs = env.ml_inputs()
        agent.learn(ml_inputs)
        for position in ml_inputs.keys():
            training_data[position].append(ml_inputs[position])
    agent.save_weights(os.path.join(training_params['save_dir'],training_params['agent_name']))
    return training_data