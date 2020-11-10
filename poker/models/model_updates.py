import numpy as np
import torch
import torch.nn.functional as F
from models.model_utils import scale_rewards,soft_update,return_value_mask
import logging
from prettytable import PrettyTable

def update_critic_batch(data,local_critic,target_critic,params):
    # get the inputs; data is a list of [inputs, targets]
    device = params['device']
    trajectory, target = data.values()
    obs = trajectory['obs'].to(device)
    action = trajectory['action'].to(device)
    reward = target['reward'].to(device)
    betsize_mask = trajectory['betsize_mask'].to(device)
    action_mask = trajectory['action_mask'].to(device)
    # scaled_rewards = scale_rewards(reward,params['min_reward'],params['max_reward'])
    # Critic update
    local_values = local_critic(obs)['value']
    value_mask = return_value_mask(action)
    # TD_error = local_values[value_mask] - reward
    # critic_loss = (TD_error**2*0.5).mean()
    critic_loss = F.smooth_l1_loss(reward,local_values[value_mask],reduction='sum')
    params['critic_optimizer'].zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(local_critic.parameters(), params['gradient_clip'])
    params['critic_optimizer'].step()
    soft_update(local_critic,target_critic,device,tau=1e-1)
    post_local_values = local_critic(obs)['value']
    # Eval
    table = PrettyTable(["Critic Values","Updated Critic values","Max diff"])
    for i in range(local_values.size(0)):
        critic_diff = local_values[i].detach().cpu().numpy() - post_local_values[i].detach().cpu().numpy()
        critic_diff_max = np.max(np.abs(critic_diff))
        table.add_row([local_values.detach()[i],post_local_values.detach()[i],critic_diff_max])
    print(table)
    table = PrettyTable(["Critic Value","Updated value","Reward","action","Loss"])
    for i in range(local_values.size(0)):
        table.add_row([local_values.detach()[i][action[i]],post_local_values.detach()[i][action[i]],reward[i],action[i],critic_loss.item()])
    print(table)
    return critic_loss.item()

def update_actor_batch(data,local_actor,target_actor,target_critic,params):
    # get the inputs; data is a list of [inputs, targets]
    device = params['device']
    trajectory, target = data.values()
    state = trajectory['state'].to(device)
    obs = trajectory['obs'].to(device)
    action = trajectory['action'].to(device)
    reward = target['reward'].to(device)
    betsize_mask = trajectory['betsize_mask'].to(device)
    action_mask = trajectory['action_mask'].to(device)
    # scaled_rewards = scale_rewards(reward,params['min_reward'],params['max_reward'])
    # Actor update #
    target_values = target_critic(obs)['value']
    actor_out = local_actor(state,action_mask,betsize_mask)
    target_out = target_actor(state,action_mask,betsize_mask)
    actor_value_mask = return_value_mask(actor_out['action'])
    expected_value = (actor_out['action_probs'].view(-1) * target_values.view(-1)).view(actor_value_mask.size()).detach().sum(-1)
    advantages = (target_values[actor_value_mask] - expected_value).view(-1)
    policy_loss = (-actor_out['action_prob'].view(-1) * advantages).sum()
    params['actor_optimizer'].zero_grad()
    policy_loss.backward()
    # torch.nn.utils.clip_grad_norm_(local_actor.parameters(), params['gradient_clip'])
    params['actor_optimizer'].step()
    soft_update(local_actor,target_actor,device)
    post_actor_out = local_actor(state,action_mask,betsize_mask)
    post_target_out = target_actor(state,action_mask,betsize_mask)
    # Assert probabilities aren't changing more than x
    actor_diff = actor_out['action_probs'].detach().cpu().numpy() - post_actor_out['action_probs'].detach().cpu().numpy()
    target_diff = target_out['action_probs'].detach().cpu().numpy() - post_target_out['action_probs'].detach().cpu().numpy()
    actor_diff_max = np.max(np.abs(actor_diff))
    target_diff_max = np.max(np.abs(target_diff))
    table = PrettyTable(["Critic Q Values","Action","Reward","Policy Loss"])
    for i in range(target_values.size(0)):
        table.add_row([target_values.detach()[i],action[i],reward[i],policy_loss.item()])
    print(table)
    table = PrettyTable(["Actor Probs","Updated Actor Probs","Max Actor diff"])
    for i in range(actor_out['action_probs'].detach().size(0)):
        table.add_row([actor_out['action_probs'].detach()[i],post_actor_out['action_probs'].detach()[i],actor_diff_max])
    print(table)
    table = PrettyTable(["Target Actor Probs","Updated Target Probs","Max Target diff"])
    for i in range(target_out['action_probs'].detach().size(0)):
        table.add_row([target_out['action_probs'].detach()[i],post_target_out['action_probs'].detach()[i],target_diff_max])
    print(table)
    return policy_loss.item()

def update_actor_critic_batch(data,local_actor,local_critic,target_actor,target_critic,params):
    # get the inputs; data is a list of [inputs, targets]
    device = params['device']
    trajectory, target = data.values()
    state = trajectory['state']
    obs = trajectory['obs']
    action = trajectory['action']
    reward = target['reward']
    betsize_mask = trajectory['betsize_mask']
    action_mask = trajectory['action_mask']
    # scaled_rewards = scale_rewards(reward,params['min_reward'],params['max_reward'])
    # Critic update
    local_values = local_critic(obs.to(device))['value']
    # target_values = target_critic(obs)['value']
    value_mask = return_value_mask(action.to(device))
    # TD_error = local_values[value_mask] - reward
    # critic_loss = (TD_error**2*0.5).mean()
    critic_loss = F.smooth_l1_loss(reward.to(device),local_values[value_mask].to(device),reduction='sum')
    params['critic_optimizer'].zero_grad()
    critic_loss.backward()
    # torch.nn.utils.clip_grad_norm_(local_critic.parameters(), params['gradient_clip'])
    params['critic_optimizer'].step()
    soft_update(local_critic,target_critic,device,tau=1e-1)
    # Actor update #
    # post_local_values = local_critic(obs)['value']
    post_target_values = target_critic(obs.to(device))['value']
    actor_out = local_actor(state.to(device),action_mask.to(device),betsize_mask.to(device))
    # target_out = target_actor(state,action_mask,betsize_mask)
    actor_value_mask = return_value_mask(actor_out['action'])
    expected_value = (actor_out['action_probs'].view(-1) * post_target_values.view(-1)).view(actor_value_mask.size()).detach().sum(-1)
    advantages = (post_target_values[actor_value_mask.to(device)] - expected_value).view(-1)
    policy_loss = (-actor_out['action_prob'].view(-1) * advantages).sum()
    params['actor_optimizer'].zero_grad()
    policy_loss.backward()
    # torch.nn.utils.clip_grad_norm_(local_actor.parameters(), params['gradient_clip'])
    params['actor_optimizer'].step()
    soft_update(local_actor,target_actor,device)
    # Check action probs and critic vals
    # post_actor_out = local_actor(state,action_mask,betsize_mask)
    # post_target_out = target_actor(state,action_mask,betsize_mask)
    # # Assert probabilities aren't changing more than x
    # actor_diff = actor_out['action_probs'].detach().cpu().numpy() - post_actor_out['action_probs'].detach().cpu().numpy()
    # target_diff = target_out['action_probs'].detach().cpu().numpy() - post_target_out['action_probs'].detach().cpu().numpy()
    # actor_diff_max = np.max(np.abs(actor_diff))
    # target_diff_max = np.max(np.abs(target_diff))
    # table = PrettyTable(["Critic Values","Updated critic values","Critic Loss"])
    # for i in range(post_target_values.size(0)):
    #     table.add_row([local_values.detach()[i].cpu(),post_local_values.detach()[i].cpu(),critic_loss.item()])
    # print(table)
    # table = PrettyTable(["Target Critic Values","Updated Target critic values","Action","Reward"])
    # for i in range(post_target_values.size(0)):
    #     table.add_row([target_values.detach()[i].cpu(),post_target_values.detach()[i].cpu(),action[i],reward[i]])
    # print(table)
    # table = PrettyTable(["Critic Q Values","Action","Reward","Policy Loss"])
    # for i in range(post_target_values.size(0)):
    #     table.add_row([post_target_values.detach().cpu()[i][action[i]],action[i],reward[i],policy_loss.item()])
    # print(table)
    # table = PrettyTable(["Actor Probs","Updated Actor Probs","Max Actor diff"])
    # for i in range(actor_out['action_probs'].detach().size(0)):
    #     table.add_row([actor_out['action_probs'].detach().cpu()[i],post_actor_out['action_probs'].detach().cpu()[i],actor_diff_max])
    # print(table)
    # table = PrettyTable(["Target Actor Probs","Updated Target Probs","Max Target diff"])
    # for i in range(target_out['action_probs'].detach().size(0)):
    #     table.add_row([target_out['action_probs'].detach().cpu()[i],post_target_out['action_probs'].detach().cpu()[i],target_diff_max])
    # print(table)
    return critic_loss.item()

def update_actor(poker_round,actor,target_actor,target_critic,params):
    """With critic batch updates"""
    actor_optimizer = params['actor_optimizer']
    device = params['device']
    state = poker_round['state']
    obs = poker_round['obs']
    action = poker_round['action']
    reward = poker_round['reward']
    betsize_mask = poker_round['betsize_mask']
    action_mask = poker_round['action_mask']
    # Actor update #
    target_values = target_critic(obs)['value']
    actor_out = actor(np.array(state),np.array(action_mask),np.array(betsize_mask))
    target_out = target_actor(np.array(state),np.array(action_mask),np.array(betsize_mask))
    actor_value_mask = return_value_mask(actor_out['action'])
    expected_value = (actor_out['action_probs'].view(-1) * target_values.view(-1)).view(actor_value_mask.size()).detach().sum(-1)
    advantages = (target_values[actor_value_mask] - expected_value).view(-1)
    policy_loss = (-actor_out['action_prob'].view(-1) * advantages).sum()
    actor_optimizer.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), params['gradient_clip'])
    actor_optimizer.step()
    soft_update(actor,target_actor,device)
    post_actor_out = actor(np.array(state),np.array(action_mask),np.array(betsize_mask))
    post_target_out = target_actor(np.array(state),np.array(action_mask),np.array(betsize_mask))
    # Assert probabilities aren't changing more than x
    actor_diff = actor_out['action_probs'].detach().cpu().numpy() - post_actor_out['action_probs'].detach().cpu().numpy()
    target_diff = target_out['action_probs'].detach().cpu().numpy() - post_target_out['action_probs'].detach().cpu().numpy()
    actor_diff_max = np.max(np.abs(actor_diff))
    target_diff_max = np.max(np.abs(target_diff))
    table = PrettyTable(["Critic Q Values","Action","Reward","Policy Loss"])
    table.add_row([target_values.detach(),action,reward,policy_loss.item()])
    print(table)
    table = PrettyTable(["Actor Probs","Updated Actor Probs","Max Actor diff"])
    table.add_row([actor_out['action_probs'].detach(),post_actor_out['action_probs'].detach(),actor_diff_max])
    print(table)
    table = PrettyTable(["Target Actor Probs","Updated Target Probs","Max Target diff"])
    table.add_row([target_out['action_probs'].detach(),post_target_out['action_probs'].detach(),target_diff_max])
    print(table)

def update_critic(poker_round,critic,params):
    critic_optimizer = params['critic_optimizer'] 
    device = params['device'] 
    state = poker_round['state']
    obs = poker_round['obs']
    action = poker_round['action']
    reward = torch.tensor(poker_round['reward']).unsqueeze(-1).to(device)
    betsize_mask = poker_round['betsize_mask']
    action_mask = poker_round['action_mask']
    ## Critic update ##
    local_values = critic(obs)['value']
    value_mask = return_value_mask(action)
    # TD_error = local_values[value_mask] - reward
    # critic_loss = (TD_error**2*0.5).mean()
    critic_loss = F.smooth_l1_loss(reward,local_values[value_mask],reduction='sum')
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()
    print('local_values[value_mask],reward',local_values[value_mask],reward)
    return critic_loss.item()

def update_combined(poker_round,model,params):
    optimizer = params['model_optimizer']  
    device = params['device']  
    state = poker_round['state']
    action = poker_round['action']
    reward = torch.tensor(poker_round['reward']).unsqueeze(-1).to(device)
    betsize_mask = poker_round['betsize_mask']
    action_mask = poker_round['action_mask']
    # scaled_rewards = scale_rewards(reward,params['min_reward'],params['max_reward'])
    ## Critic update ##
    local_values = model(np.array(state),np.array(action_mask),np.array(betsize_mask))['value']
    value_mask = return_value_mask(action)
    # TD_error = local_values[value_mask] - reward
    # critic_loss = (TD_error**2*0.5).sum()
    critic_loss = F.smooth_l1_loss(reward,local_values[value_mask],reduction='sum')
    optimizer.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), params['gradient_clip'])
    optimizer.step()
    # Actor update #
    actor_out = model(np.array(state),np.array(action_mask),np.array(betsize_mask))
    target_values = actor_out['value']
    actor_value_mask = return_value_mask(actor_out['action'])
    expected_value = (actor_out['action_probs'].view(-1) * target_values.view(-1)).view(actor_value_mask.size()).detach().sum(-1)
    advantages = (target_values[actor_value_mask] - expected_value).view(-1)
    policy_loss = (-actor_out['action_prob'].view(-1) * advantages).sum()
    optimizer.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), params['gradient_clip'])
    optimizer.step()
    return critic_loss.item(),policy_loss.item()

def update_actor_critic(poker_round,critic,target_critic,actor,target_actor,params):
    critic_optimizer = params['critic_optimizer']
    actor_optimizer = params['actor_optimizer']
    device = params['device']
    state = poker_round['state']
    obs = poker_round['obs']
    action = poker_round['action']
    reward = torch.tensor(poker_round['reward']).unsqueeze(-1).to(device)
    betsize_mask = poker_round['betsize_mask']
    action_mask = poker_round['action_mask']
    ## Critic update ##
    # local_values = critic(obs)['value']
    # value_mask = return_value_mask(action)
    # # TD_error = local_values[value_mask] - reward
    # # critic_loss = (TD_error**2*0.5).mean()
    # critic_loss = F.smooth_l1_loss(reward,local_values[value_mask],reduction='sum')
    # critic_optimizer.zero_grad()
    # critic_loss.backward()
    # critic_optimizer.step()
    # Actor update #
    target_values = target_critic(obs)['value']
    actor_out = actor(state,action_mask,betsize_mask)
    actor_value_mask = return_value_mask(actor_out['action'])
    expected_value = (actor_out['action_probs'].view(-1) * target_values.view(-1)).view(actor_value_mask.size()).detach().sum(-1)
    advantages = (target_values[actor_value_mask] - expected_value).view(-1)
    policy_loss = (-actor_out['action_prob'].view(-1) * advantages).sum()
    actor_optimizer.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), params['gradient_clip'])
    actor_optimizer.step()
    # print('\nprobs,prob,actor action,original action',actor_out['action_probs'].detach(),actor_out['action_prob'].detach(),actor_out['action'],action)
    # print('\nlocal_values,Q_value',local_values,local_values[value_mask].item())
    # print('\ntarget_values,target_Q_value',target_values,target_values[value_mask].item())
    # print('\ntarget_values*mask',(actor_out['action_probs'].view(-1) * target_values.view(-1)).view(value_mask.size()))
    # print('\nexpected_value',expected_value)
    # print('\nadvantages',advantages)
    # print('\nreward',reward)
    # print('\npolicy_loss',policy_loss)