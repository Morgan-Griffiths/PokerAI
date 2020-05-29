
import os
import random
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable as V
from torch import optim
import copy

from models.networks import Baseline,Dueling_QNetwork,HoldemBaselineCritic,HoldemBaseline,BaselineKuhnCritic,BaselineCritic
from models.model_utils import hard_update
from models.buffers import PriorityReplayBuffer
import kuhn.datatypes as pdt
import poker.datatypes as hdt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def return_agent(agent_type,nS,nO,nA,nB,seed,params):
    if agent_type == pdt.AgentTypes.ACTOR:
        agent = BaselineAgent(nS,nO,nA,nB,seed,params)
    elif agent_type == pdt.AgentTypes.ACTOR_CRITIC:
        agent = Agent(nS,nO,nA,nB,seed,params)
    elif agent_type == pdt.AgentTypes.COMBINED_ACTOR_CRITIC:
        agent = CombinedAgent(nS,nO,nA,nB,seed,params)
    elif agent_type == hdt.AgentTypes.SINGLE or agent_type == hdt.AgentTypes.SPLIT or agent_type == hdt.AgentTypes.SPLIT_OBS:
        agent = FullAgent(nS,nO,nA,nB,seed,params)
    else:
        raise ValueError(f'Agent not supported {agent_type}')
    return agent

class BaselineAgent(object):
    def __init__(self,nS,nO,nA,nB,seed,params):
        super().__init__()
        self.nS = nS
        self.nO = nO
        self.nA = nA
        self.seed = seed
        self.network = params['network'](seed,nS,nA,params)
        self.optimizer = optim.Adam(self.network.parameters(), lr=1e-4)
        
    def __call__(self,*args):
        return self.network(*args)
    
    def learn(self,player_data):
        positions = player_data.keys()
        for position in positions:
            action_probs = player_data[position]['action_probs']
            game_states = player_data[position]['game_states']
            rewards = player_data[position]['rewards']
            # actions = player_data[position]['actions']
            # observations = player_data[position]['observations']
            # dones = player_data[position]['dones']
            # indexes = player_data[position]['indexes']
            if len(game_states):
                self.backward(action_probs,rewards)
            
    def backward(self,action_probs,rewards):
        policy_loss = (-action_probs * rewards).sum()
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

    def load_weights(self,path):
        self.network.load_state_dict(torch.load(path))
        self.network.eval()

    def save_weights(self,path):
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.mkdir(directory)
        torch.save(self.network.state_dict(), path)

class Agent(object):
    def __init__(self,nS,nO,nA,nB,seed,params):
        super().__init__()
        self.nS = nS
        self.nO = nO
        self.nA = nA
        self.nB = nB
        self.nC = nA - 2 + nB
        self.seed = seed
        self.epochs = params['epochs']+1
        self.network_output = params['network_output']
        self.tau = params['TAU']
        self.historical_states = params['historical_states']
        self.max_reward = params['max_reward']
        self.min_reward = params['min_reward']
        self.gradient_clip = params['CLIP_NORM']
        self.critic_type = params['critic_type']
        self.local_actor = params['actor_network'](seed,nS,nA,nB,params)
        self.target_actor = params['actor_network'](seed,nS,nA,nB,params)
        self.local_critic = params['critic_network'](seed,nO,nA,nB,params)
        self.target_critic = params['critic_network'](seed,nO,nA,nB,params)
        self.target_critic.eval()
        print('critic_type',self.critic_type)
        if self.critic_type == 'q' or self.critic_type == 'single':
            self.critic_backward = self.qcritic_backward
            self.actor_backward = self.qactor_backward
            self.critique = self.qcritique
        else:
            self.critic_backward = self.reg_critic_backward
            self.actor_backward = self.reg_actor_backward
            self.critique = self.reg_critique

        if self.historical_states == True:
            self.actor_lr = 1e-4
            self.critic_lr = 1e-4
        else:
            self.actor_lr = 1e-4
            self.critic_lr = 1e-4
        print(f"Agent learning rates: Critic {self.actor_lr}, Actor {self.critic_lr}")

        self.actor_optimizer = optim.Adam(self.local_actor.parameters(), lr=self.actor_lr,weight_decay=params['L2'])
        self.critic_optimizer = optim.Adam(self.local_critic.parameters(), lr=self.critic_lr)
        # Copy the weights from local to target
        hard_update(self.local_critic,self.target_critic)
        hard_update(self.local_actor,self.target_actor)
        
    def __call__(self,*args):
        return self.local_actor(*args)

    def reg_critique(self,obs,action):
        return self.target_critic(obs,action)

    def qcritique(self,obs,action):
        return self.target_critic(obs)
    
    def learn(self,player_data):
        positions = player_data.keys()
        for position in positions:
            action_prob = player_data[position]['action_prob']
            action_probs = player_data[position]['action_probs']
            game_states = player_data[position]['game_states']
            rewards = player_data[position]['rewards']
            actions = player_data[position]['actions'].view(-1)
            observations = player_data[position]['observations']
            action_masks = player_data[position]['action_masks']
            # print(position)
            # print('game_states',game_states)
            # print('actions',actions)
            # print('rewards',rewards)
            # print('action_probs',action_probs)
            critic_inputs = {
                'rewards':rewards, 
                'game_states':game_states,
                'observations':observations,
                'actions':actions
            }
            actor_inputs = {
                'actions':actions,
                'action_prob':action_prob,
                'action_probs':action_probs,
                'game_states':game_states,
                'action_masks':action_masks
            }
            if self.historical_states == True:
                critic_inputs['historical_game_states'] = player_data[position]['historical_game_states']
            if 'betsizes' in player_data[position]:
                betsizes = player_data[position]['betsizes']
                betsize_prob = player_data[position]['betsize_prob']
                betsize_probs = player_data[position]['betsize_probs']
                betsize_masks = player_data[position]['betsize_masks']
                critic_inputs['betsizes'] = betsizes
                critic_inputs['betsize_prob'] = betsize_prob
                critic_inputs['betsize_probs'] = betsize_probs
                actor_inputs['betsize_masks'] = betsize_masks
            # dones = player_data[position]['dones']
            # indexes = player_data[position]['indexes']
            if len(game_states):
                # for _ in range(2):
                if self.historical_states == True:
                    self.combo_backward(actor_inputs,critic_inputs)
                else:
                    self.critic_backward(critic_inputs)
                    self.actor_backward(actor_inputs,critic_inputs)
            # del critic_inputs,actor_inputs
            # del action_prob,action_probs,game_states,rewards,actions,observations,action_masks

    def reg_critic_backward(self,critic_inputs:dict):
        """Performs critic update. optionally has betsizes for updating the betsize portion of critic"""
        values = self.local_critic(critic_inputs['observations'],critic_inputs['actions'])['value']
        scaled_rewards = critic_inputs['rewards']/self.max_reward
        critic_loss = F.smooth_l1_loss(scaled_rewards,values)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.local_critic.parameters(), self.gradient_clip)
        self.critic_optimizer.step()
        Agent.soft_update(self.local_critic,self.target_critic,self.tau)
            
    def reg_actor_backward(self,actor_inputs,critic_inputs):
        values = self.target_critic(critic_inputs['observations'],critic_inputs['actions'])['value']
        policy_loss = (-actor_inputs['action_prob'] * values).sum()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.local_actor.parameters(), self.gradient_clip)
        self.actor_optimizer.step()
        Agent.soft_update(self.local_actor,self.target_actor,self.tau)

    def return_value_mask(self,actions):
        """Returns a mask that indexes Q values by the action taken"""
        M = actions.size(0)
        if self.network_output == 'flat':
            value_mask = torch.zeros(M,self.nC)
        else:
            value_mask = torch.zeros(M,self.nA)
        if actions.dim() > 1:
            actions = actions.squeeze(1)
        value_mask[torch.arange(M),actions] = 1
        return value_mask.bool()

    def return_bet_mask(self,actions):
        """Returns a mask that indexes actions by whether there was a bet or raise"""
        mask = actions.gt(2).view(-1)
        return mask

    def scale_rewards(self,rewards,factor=1):
        """Scales rewards between -1 and 1, with optional factor to increase valuation differences"""
        return (2 * ((rewards + self.min_reward) / (self.max_reward + self.min_reward)) - 1) * factor

    def loop_critic(self,states):
        values = []
        target_values = []
        for batch in range(states.size(0)):
            value = self.local_critic(states[batch,:,:])['value']
            target_value = self.target_critic(states[batch,:,:])['value']
            values.append(value)
            target_values.append(target_value)
        values = torch.stack(values).squeeze(1)
        target_values = torch.stack(target_values).squeeze(1)
        return values,target_values

    def loop_actor(self,states,action_masks,betsize_masks):
        out = []
        for batch in range(states.size(0)):
            actor_out = self.local_actor(states[batch,:,:],action_masks[batch,:],betsize_masks[batch,:])['action_probs']
            out.append(actor_out)
        out = torch.stack(out).squeeze(1)
        return out

    def combo_backward(self,actor_inputs:dict,critic_inputs:dict):
        """Updates critic and actor at the same time"""
        ## Critic update ##
        values,target_values = self.loop_critic(critic_inputs['historical_game_states'])
        value_mask = self.return_value_mask(critic_inputs['actions'])
        scaled_rewards = self.scale_rewards(critic_inputs['rewards'])
        critic_loss = F.smooth_l1_loss(scaled_rewards.view(value_mask.size(0)),values[value_mask],reduction='sum')
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.local_critic.parameters(), self.gradient_clip)
        self.critic_optimizer.step()
        Agent.soft_update(self.local_critic,self.target_critic,self.tau)

        ## Actor update ##
        # prob update
        expected_value = (actor_inputs['action_probs'].view(-1) * target_values.view(-1)).detach().sum(-1)
        advantages = target_values[value_mask] - expected_value
        policy_loss = (-actor_inputs['action_prob'].view(-1) * advantages).sum()
        # Label update TODO Smooth label update
        # 1 hot update
        # hot_encoder = torch.nn.functional.one_hot(torch.arange(0,self.nC))
        # labels = hot_encoder[]
        # combined_mask = torch.cat((actor_inputs['action_masks'][:,:3],actor_inputs['betsize_masks'][:,:]),dim=-1)
        # labels = torch.argmax(target_values * combined_mask,dim=-1)
        # policy_loss = F.nll_loss(actor_inputs['action_probs'],labels)
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.local_actor.parameters(), self.gradient_clip)
        self.actor_optimizer.step()
        Agent.soft_update(self.local_actor,self.target_actor,self.tau)
        # Assert learning
        # pre_value = values[value_mask]
        # pre_values = values# * actor_inputs['betsize_masks']
        # post_values,_ = self.loop_critic(critic_inputs['historical_game_states'])
        # post_value = post_values[value_mask]
        # pre_probs = actor_inputs['action_probs']
        # post_probs = self.loop_actor(critic_inputs['historical_game_states'],actor_inputs['action_masks'],actor_inputs['betsize_masks'])
        # print('post')

    def qcritic_backward(self,critic_inputs:dict):
        """Computes critic grad update. Optionally computes betsize grad update in unison"""
        values = self.local_critic(critic_inputs['game_states'])['value']
        scaled_rewards = self.scale_rewards(critic_inputs['rewards'])
        value_mask = self.return_value_mask(critic_inputs['actions'])
        critic_loss = F.smooth_l1_loss(scaled_rewards.view(value_mask.size(0)),values[value_mask],reduction='sum')
        self.critic_optimizer.zero_grad()
        if 'betsize' in critic_output:
            betsize_categories = critic_inputs['betsizes']
            bet_mask = self.return_bet_mask(critic_inputs['actions'])
            all_betsize_values = critic_output['betsize'][bet_mask]
            real_betsize_categories = betsize_categories[bet_mask].view(-1)
            row = torch.arange(real_betsize_categories.size(0))
            betsize_values = all_betsize_values[row,real_betsize_categories].unsqueeze(1)
            betsize_rewards = scaled_rewards[bet_mask]
            if betsize_rewards.size(0) > 0:
                betsize_loss = F.smooth_l1_loss(betsize_values,betsize_rewards)
                betsize_loss.backward(retain_graph=True)
                print('betsize_loss',betsize_loss)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.local_critic.parameters(), self.gradient_clip)
        self.critic_optimizer.step()
        Agent.soft_update(self.local_critic,self.target_critic,self.tau)
        # Assert learning
        # pre_value = critic_output['value'][value_mask]
        # critic_output = self.local_critic(critic_inputs['game_states'])
        # post_value = critic_output['value'][value_mask]
        # print('post update values')
            
    def qactor_backward(self,actor_inputs:dict,critic_inputs:dict):
        """
        critic_inputs: rewards,obs,actions,betsizes,betsize_prob,betsize_probs
        critic_outputs: value,betsize (Q values over action categories and betsize categories)
        actor_inputs: actions,action_prob,action_probs,game_states
        actor_outputs: action,action_prob,action_probs,betsize,betsize_prob,betsize_probs
        """
        critic_outputs = self.target_critic(critic_inputs['game_states'])
        value_mask = self.return_value_mask(actor_inputs['actions'])
        values = critic_outputs['value']
        expected_value = (actor_inputs['action_probs'].view(-1) * values.view(-1)).detach().sum(-1)
        advantages = values[value_mask] - expected_value
        policy_loss = (-actor_inputs['action_prob'].view(-1) * advantages).sum()
        self.actor_optimizer.zero_grad()
        # select all instances of bets
        if 'betsize' in critic_outputs:
            bet_mask = self.return_bet_mask(actor_inputs['actions'])
            if True in bet_mask:
                all_betsize_values = critic_inputs['betsizes'][bet_mask]
                betsize_probs = critic_inputs['betsize_probs'][bet_mask]
                betsize_prob = critic_inputs['betsize_prob'][bet_mask]
                # isolate the bet values
                real_betsize_categories = all_betsize_values.view(-1)
                row = torch.arange(real_betsize_categories.size(0))
                betsize_values = critic_outputs['betsize'][row,real_betsize_categories]
                masked_betsize_values = critic_outputs['betsize'] * actor_inputs['betsize_masks'].long()
                betsize_expected_value = (betsize_probs * masked_betsize_values[bet_mask]).detach().sum(-1)
                betsize_advantages = betsize_values - betsize_expected_value
                betsize_policy_loss = (-betsize_prob.view(-1) * betsize_advantages).sum()
                betsize_policy_loss.backward(retain_graph=True)

        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.local_actor.parameters(), self.gradient_clip)
        self.actor_optimizer.step()
        Agent.soft_update(self.local_actor,self.target_actor,self.tau)
        # pre_probs = actor_inputs['action_probs']
        # actor_out = self.local_actor(actor_inputs['game_states'],actor_inputs['action_masks'],actor_inputs['betsize_masks'])
        # post_probs = actor_out['action_probs']
        # print('post')

    def load_weights(self,path):
        self.local_actor.load_state_dict(torch.load(path + '_actor'))
        self.local_critic.load_state_dict(torch.load(path + '_critic'))
        self.local_actor.eval()
        self.local_critic.eval()

    def save_weights(self,path):
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.mkdir(directory)
        torch.save(self.local_actor.state_dict(), path + '_actor')
        torch.save(self.local_critic.state_dict(), path + '_critic')

    def update_networks(self):
        self.target_critic = Agent.soft_update_target(self.local_critic,self.target_critic,self.tau)
        self.target_actor = Agent.soft_update_target(self.local_actor,self.target_actor,self.tau)

    @staticmethod
    def soft_update(local,target,tau):
        for local_param,target_param in zip(local.parameters(),target.parameters()):
            target_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)
        
class ParallelAgent(Agent):
    def __init__(self,nS,nO,nA,nB,seed,params,actor,critic):
        super(ParallelAgent).__init__()
        self.nS = nS
        self.nO = nO
        self.nA = nA
        self.nB = nB
        self.nC = nA - 2 + nB
        self.seed = seed
        self.epochs = params['epochs']+1
        self.network_output = params['network_output']
        self.tau = params['TAU']
        if 'historical_states' in params:
            self.historical_states = params['historical_states']
        self.max_reward = params['max_reward']
        self.min_reward = params['min_reward']
        self.gradient_clip = params['CLIP_NORM']
        self.critic_type = params['critic_type']
        self.local_actor = actor
        self.target_actor = params['actor_network'](seed,nS,nA,nB,params)
        self.local_critic = critic
        self.target_critic = params['critic_network'](seed,nS,nA,nB,params)
        print('critic_type',self.critic_type)
        if self.critic_type == 'q' or self.critic_type == 'single':
            self.critic_backward = self.qcritic_backward
            self.actor_backward = self.qactor_backward
            self.critique = self.qcritique
        else:
            self.critic_backward = self.reg_critic_backward
            self.actor_backward = self.reg_actor_backward
            self.critique = self.reg_critique

        if self.historical_states == True:
            self.actor_lr = 1e-4
            self.critic_lr = 1e-4
        else:
            self.actor_lr = 1e-4
            self.critic_lr = 1e-4
        print(f"Agent learning rates: Critic {self.actor_lr}, Actor {self.critic_lr}")

        self.actor_optimizer = optim.Adam(self.local_actor.parameters(), lr=self.actor_lr,weight_decay=params['L2'])
        self.critic_optimizer = optim.Adam(self.local_critic.parameters(), lr=self.critic_lr)
        # Copy the weights from local to target
        hard_update(self.local_critic,self.target_critic)
        hard_update(self.local_actor,self.target_actor)

class CombinedAgent(Agent):
    def __init__(self,nS,nO,nA,nB,seed,params):
        super(CombinedAgent).__init__()
        self.nS = nS
        self.nO = nO
        self.nA = nA
        self.nB = nB
        self.nC = nA - 2 + nB
        self.seed = seed
        self.epochs = params['epochs']+1
        self.network_output = params['network_output']
        self.tau = params['TAU']
        self.max_reward = params['max_reward']
        self.min_reward = params['min_reward']
        self.gradient_clip = params['CLIP_NORM']
        self.critic_type = params['critic_type']
        self.local_network = params['combined_network'](seed,nS,nA,nB,params)
        self.target_network = params['combined_network'](seed,nS,nA,nB,params)
        self.network_optimizer = optim.Adam(self.local_network.parameters(), lr=5e-3,weight_decay=params['L2'])
        # Copy the weights from local to target
        hard_update(self.local_network,self.target_network)

    def __call__(self,*args):
        return self.local_network(*args)

    def learn(self,player_data):
        positions = player_data.keys()
        for position in positions:
            action_prob = player_data[position]['action_prob']
            action_probs = player_data[position]['action_probs']
            game_states = player_data[position]['game_states']
            rewards = player_data[position]['rewards']
            actions = player_data[position]['actions'].view(-1)
            observations = player_data[position]['observations']
            action_masks = player_data[position]['action_masks']
            # print(position)
            # print('game_states',game_states)
            # print('actions',actions)
            # print('rewards',rewards)
            # print('action_probs',action_probs)
            critic_inputs = {
                'rewards':rewards, 
                'observations':observations,
                'game_states':game_states,
                'actions':actions,
                'action_masks':action_masks
            }
            actor_inputs = {
                'actions':actions,
                'action_prob':action_prob,
                'action_probs':action_probs,
                'game_states':game_states,
                'action_masks':action_masks
            }
            if 'betsizes' in player_data[position]:
                betsizes = player_data[position]['betsizes']
                betsize_prob = player_data[position]['betsize_prob']
                betsize_probs = player_data[position]['betsize_probs']
                betsize_masks = player_data[position]['betsize_masks']
                critic_inputs['betsizes'] = betsizes
                critic_inputs['betsize_prob'] = betsize_prob
                critic_inputs['betsize_probs'] = betsize_probs
                actor_inputs['betsize_masks'] = betsize_masks
                critic_inputs['betsize_masks'] = betsize_masks
            # dones = player_data[position]['dones']
            # indexes = player_data[position]['indexes']
            if len(game_states):
                self.combined_backward(actor_inputs,critic_inputs)

    def combined_backward(self,actor_inputs:dict,critic_inputs:dict):
        """Computes critic grad update. Optionally computes betsize grad update in unison"""
        target_outputs = self.target_act(critic_inputs['game_states'],critic_inputs['action_masks'],critic_inputs['betsize_masks'])
        value_mask = self.return_value_mask(critic_inputs['actions'])
        scaled_rewards = self.scale_rewards(critic_inputs['rewards'])
        values = target_outputs['value']
        critic_loss = F.smooth_l1_loss(scaled_rewards.view(value_mask.size(0)),values[value_mask])
        expected_value = (actor_inputs['action_probs'].view(-1) * values.view(-1)).detach().sum(-1)
        advantages = values[value_mask] - expected_value
        policy_loss = (-actor_inputs['action_prob'].view(-1) * advantages).sum()
        self.network_optimizer.zero_grad()
        if 'betsize' in critic_output:
            betsize_categories = critic_inputs['betsizes']
            bet_mask = self.return_bet_mask(critic_inputs['actions'])
            all_betsize_values = critic_output['betsize'][bet_mask]
            real_betsize_categories = betsize_categories[bet_mask].view(-1)
            row = torch.arange(real_betsize_categories.size(0))
            betsize_values = all_betsize_values[row,real_betsize_categories].unsqueeze(1)
            betsize_rewards = scaled_rewards[bet_mask]
            if betsize_rewards.size(0) > 0:
                betsize_loss = F.smooth_l1_loss(betsize_values,betsize_rewards)
                betsize_loss.backward(retain_graph=True)
                print('betsize_loss',betsize_loss)
        (critic_loss+policy_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.local_network.parameters(), self.gradient_clip)
        self.network_optimizer.step()
        Agent.soft_update(self.local_network,self.target_network,self.tau)
        # Assert learning
        # pre_value = values[value_mask]
        # critic_output = self.local_network(critic_inputs['game_states'],critic_inputs['action_masks'],critic_inputs['betsize_masks'])
        # value_mask = self.return_value_mask(critic_inputs['actions'])
        # scaled_rewards = (critic_inputs['rewards']/self.max_reward).squeeze(1)
        # post_value = critic_output['value'][value_mask]
        # print('post update values')

####################################
###         Full Poker           ###
####################################

class FullAgent(Agent):
    def __init__(self,nS,nO,nA,nB,seed,params,actor=None,critic=None):
        super(FullAgent).__init__()
        self.nS = nS
        self.nO = nO
        self.nA = nA
        self.nB = nB
        self.nC = nA - 2 + nB
        self.seed = seed
        self.epochs = params['epochs']+1
        self.network_output = params['network_output']
        self.tau = params['TAU']
        self.max_reward = params['max_reward']
        self.min_reward = params['min_reward']
        self.gradient_clip = params['CLIP_NORM']
        self.critic_type = params['critic_type']
        if actor == None:
            self.local_actor = params['actor_network'](seed,nS,nA,nB,params)
            self.local_critic = params['critic_network'](seed,nO,nA,nB,params)
        else:
            self.local_actor = actor
            self.local_critic = critic
        self.target_actor = params['actor_network'](seed,nS,nA,nB,params)
        self.target_critic = params['critic_network'](seed,nS,nA,nB,params)
        self.target_critic.eval()
        if params['frozen_layer'] == True:
            self.update_weights(params)
        print('critic_type',self.critic_type)
        if self.critic_type == 'q' or self.critic_type == 'single':
            self.critic_backward = self.qcritic_backward
            self.actor_backward = self.qactor_backward
            self.critique = self.qcritique
        else:
            self.critic_backward = self.reg_critic_backward
            self.actor_backward = self.reg_actor_backward
            self.critique = self.reg_critique
        self.actor_lr = params['actor_lr']
        self.critic_lr = params['critic_lr']
        print(f"Agent learning rates: Critic {self.critic_lr}, Actor {self.actor_lr}")

        self.actor_optimizer = optim.Adam(self.local_actor.parameters(), lr=self.actor_lr,weight_decay=params['L2'])
        self.critic_optimizer = optim.Adam(self.local_critic.parameters(), lr=self.critic_lr)
        # Copy the weights from local to target
        hard_update(self.local_critic,self.target_critic)
        hard_update(self.local_actor,self.target_actor)
            
    def update_weights(self,params):
        layer_weights = torch.load(params['frozen_layer_path'])
        for name, param in self.local_critic.process_input.hand_board.named_parameters():
            param.data.copy_(layer_weights[name].data)
            param.requires_grad = False
        for name, param in self.local_actor.process_input.hand_board.named_parameters():
            param.data.copy_(layer_weights[name].data)
            param.requires_grad = False
    
    def learn(self,player_data):
        positions = player_data.keys()
        for position in positions:
            action_prob = player_data[position]['action_prob']
            action_probs = player_data[position]['action_probs']
            game_states = player_data[position]['game_states']
            rewards = player_data[position]['rewards']
            actions = player_data[position]['actions'].view(-1)
            observations = player_data[position]['observations']
            action_masks = player_data[position]['action_masks']
            dones = player_data[position]['dones'].view(-1)
            critic_inputs = {
                'rewards':rewards, 
                'historical_game_states':player_data[position]['historical_game_states'],
                'game_states':game_states,
                'observations':observations,
                'actions':actions,
                'dones':dones
            }
            actor_inputs = {
                'actions':actions,
                'action_prob':action_prob,
                'action_probs':action_probs,
                'game_states':game_states,
                'action_masks':action_masks
            }
            if 'betsizes' in player_data[position]:
                betsizes = player_data[position]['betsizes']
                betsize_prob = player_data[position]['betsize_prob']
                betsize_probs = player_data[position]['betsize_probs']
                betsize_masks = player_data[position]['betsize_masks']
                critic_inputs['betsizes'] = betsizes
                critic_inputs['betsize_prob'] = betsize_prob
                critic_inputs['betsize_probs'] = betsize_probs
                actor_inputs['betsize_masks'] = betsize_masks
            if len(game_states):
                self.combo_backward(actor_inputs,critic_inputs)
    
    def combo_backward(self,actor_inputs:dict,critic_inputs:dict):
        """Updates critic and actor at the same time"""
        ## Critic update ##
        local_values,target_values = self.loop_critic(critic_inputs['historical_game_states'])
        value_mask = self.return_value_mask(critic_inputs['actions'])
        scaled_rewards = self.scale_rewards(critic_inputs['rewards'])
        if len(value_mask) > 1:
            values = local_values[:-1,:]
            next_values = target_values[1:,:].detach()
            TD_error = values[value_mask[:-1]] - (next_values[value_mask[1:]] + scaled_rewards)
            TD_error += local_values[-1,:][value_mask[-1]] - scaled_rewards
        else:
            TD_error = local_values[value_mask] - scaled_rewards
        critic_loss = (TD_error**2*0.5).mean()
        # critic_loss = F.smooth_l1_loss(scaled_rewards.view(value_mask.size(0)),TD_error,reduction='sum')
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.local_critic.parameters(), self.gradient_clip)
        self.critic_optimizer.step()
        Agent.soft_update(self.local_critic,self.target_critic,self.tau)

        ## Actor update ##
        expected_value = (actor_inputs['action_probs'].view(-1) * target_values.view(-1)).view(value_mask.size()).detach().sum(-1)
        advantages = (target_values[value_mask] - expected_value).view(-1)
        policy_loss = (-actor_inputs['action_prob'].view(-1) * advantages).sum()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.local_actor.parameters(), self.gradient_clip)
        self.actor_optimizer.step()
        Agent.soft_update(self.local_actor,self.target_actor,self.tau)
        # Assert learning
        # pre_value = local_values[value_mask]
        # pre_values = local_values# * actor_inputs['betsize_masks']
        # post_values,_ = self.loop_critic(critic_inputs['historical_game_states'])
        # post_value = post_values[value_mask]
        # pre_probs = actor_inputs['action_probs']
        # post_probs = self.loop_actor(critic_inputs['historical_game_states'],actor_inputs['action_masks'],actor_inputs['betsize_masks'])
        # pass
    
    def return_value_mask(self,actions):
        """Returns a mask that indexes Q values by the action taken"""
        M = actions.size(0)
        if self.network_output == 'flat':
            value_mask = torch.zeros(M,self.nC)
        else:
            value_mask = torch.zeros(M,self.nA)
        if actions.dim() > 1:
            actions = actions.squeeze(1)
        value_mask[torch.arange(M),actions] = 1
        return value_mask.bool()

    def loop_critic(self,states):
        values = []
        target_values = []
        for batch in range(states.size(0)):
            value = self.local_critic(states[batch,:,:])['value']
            target_value = self.target_critic(states[batch,:,:])['value']
            values.append(value)
            target_values.append(target_value)
        values = torch.stack(values).squeeze(1)
        target_values = torch.stack(target_values).squeeze(1)
        return values,target_values

    def loop_actor(self,states,action_masks,betsize_masks):
        out = []
        for batch in range(states.size(0)):
            actor_out = self.local_actor(states[batch,:,:].unsqueeze(0),action_masks[batch,:],betsize_masks[batch,:])['action_probs']
            out.append(actor_out)
        out = torch.stack(out).squeeze(1)
        return out

class BetAgent(object):
    def __init__(self):
        pass

    def __call__(self,state,action_mask,betsize_mask):
        if betsize_mask.sum() > 0:
            action = torch.argmax(betsize_mask,dim=-1) + 3
        else:
            action = torch.argmax(action_mask,dim=-1)
        actor_outputs = {
            'action':action,
            'action_category':torch.argmax(action_mask,dim=-1),
            'action_probs':torch.zeros(5).fill_(2.),
            'action_prob':torch.tensor([1.]),
            'betsize' : torch.argmax(betsize_mask,dim=-1)
        }
        return actor_outputs