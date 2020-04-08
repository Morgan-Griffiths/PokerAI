import torch

mask_dict = {
        5:torch.Tensor([1,1,1,0]),
        0:torch.Tensor([0,0,0,0]),
        1:torch.Tensor([0,0,1,1]),
        2:torch.Tensor([0,0,0,0])
        }

def return_mask(state,action_index):
    return mask_dict[state[0,action_index].long().item()]

def train(env,agent,training_params):
    training_data = training_params['training_data']
    for e in range(training_params['epochs']):
        state,obs,done = env.reset()
        mask = return_mask(state,training_params['action_index'])
        while not done:
            action,log_probs = agent(state,mask)
            state,obs,done = env.step(action,log_probs)
            if state[0,-1] != -1:
                mask = return_mask(state,training_params['action_index'])
        ml_inputs = env.ml_inputs()
        agent.learn(ml_inputs)
        for position in ml_inputs.keys():
            training_data[position].append(ml_inputs[position])
    return training_data