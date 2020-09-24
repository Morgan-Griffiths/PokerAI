import numpy as np
import torch
import torch.nn.functional as F
from models.model_utils import scale_rewards,soft_update,return_value_mask

def update_critic(poker_round,critic,params):
    critic_optimizer = params['critic_optimizer']  
    state = poker_round['state']
    obs = poker_round['obs']
    action = poker_round['action']
    reward = torch.tensor(poker_round['reward']).unsqueeze(-1)
    betsize_mask = poker_round['betsize_mask']
    action_mask = poker_round['action_mask']
    ## Critic update ##
    local_values = critic(obs)['value']
    value_mask = return_value_mask(action)
    TD_error = local_values[value_mask] - reward
    # critic_loss = (TD_error**2*0.5).mean()
    critic_loss = F.smooth_l1_loss(reward,TD_error,reduction='sum')
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()
    # print('local_values',local_values)
    # print('value_mask,action',value_mask,action)
    # print('local_values[value_mask],reward',local_values[value_mask],reward)
    return critic_loss.item()

def update_combined(poker_round,model,params):
    optimizer = params['model_optimizer']  
    state = poker_round['state']
    action = poker_round['action']
    reward = poker_round['reward']
    betsize_mask = poker_round['betsize_mask']
    action_mask = poker_round['action_mask']
    scaled_rewards = scale_rewards(reward,params['min_reward'],params['max_reward'])
    ## Critic update ##
    local_values = model(np.array(state),np.array(action_mask),np.array(betsize_mask))['value']
    value_mask = return_value_mask(action)
    TD_error = local_values[value_mask] - scaled_rewards
    # critic_loss = (TD_error**2*0.5).sum()
    critic_loss = F.smooth_l1_loss(torch.tensor(scaled_rewards).unsqueeze(-1),TD_error,reduction='sum')
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
    state = poker_round['state']
    obs = poker_round['obs']
    action = poker_round['action']
    reward = torch.tensor(poker_round['reward']).unsqueeze(-1)
    betsize_mask = poker_round['betsize_mask']
    action_mask = poker_round['action_mask']
    ## Critic update ##
    local_values = critic(obs)['value']
    value_mask = return_value_mask(action)
    TD_error = local_values[value_mask] - reward
    # critic_loss = (TD_error**2*0.5).mean()
    critic_loss = F.smooth_l1_loss(reward,TD_error,reduction='sum')
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()
    soft_update(critic,target_critic)
    # Actor update #
    target_values = target_critic(obs)['value']
    actor_out = actor(np.array(state),np.array(action_mask),np.array(betsize_mask))
    actor_value_mask = return_value_mask(actor_out['action'])
    expected_value = (actor_out['action_probs'].view(-1) * target_values.view(-1)).view(actor_value_mask.size()).detach().sum(-1)
    advantages = (target_values[actor_value_mask] - expected_value).view(-1)
    policy_loss = (-actor_out['action_prob'].view(-1) * advantages).sum()
    actor_optimizer.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), params['gradient_clip'])
    actor_optimizer.step()
    soft_update(actor,target_actor)
    # print('\nprobs,prob,actor action,original action',actor_out['action_probs'].detach(),actor_out['action_prob'].detach(),actor_out['action'],action)
    # print('\nlocal_values,Q_value',local_values,local_values[value_mask].item())
    # print('\ntarget_values,target_Q_value',target_values,target_values[value_mask].item())
    # print('\ntarget_values*mask',(actor_out['action_probs'].view(-1) * target_values.view(-1)).view(value_mask.size()))
    # print('\nexpected_value',expected_value)
    # print('\nadvantages',advantages)
    # print('\nreward',reward)
    # print('\npolicy_loss',policy_loss)
    return critic_loss.item(),policy_loss.item()