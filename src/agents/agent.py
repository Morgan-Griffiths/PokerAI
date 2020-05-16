
import os
import random
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable as V
from torch import optim

from models.networks import Baseline,Dueling_QNetwork,HoldemBaselineCritic,HoldemBaseline,BaselineKuhnCritic,BaselineCritic,hard_update
from models.buffers import PriorityReplayBuffer
import poker.datatypes as pdt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def return_agent(agent_type,nS,nO,nA,nB,seed,params):
    if agent_type == pdt.AgentTypes.ACTOR:
        agent = BaselineAgent(nS,nO,nA,nB,seed,params)
    elif agent_type == pdt.AgentTypes.ACTOR_CRITIC:
        agent = Agent(nS,nO,nA,nB,seed,params)
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
        self.max_reward = params['max_reward']
        self.min_reward = params['min_reward']
        self.gradient_clip = params['CLIP_NORM']
        self.critic_type = params['critic_type']
        self.local_actor = params['actor_network'](seed,nS,nA,nB,params)
        self.target_actor = params['actor_network'](seed,nS,nA,nB,params)
        self.local_critic = params['critic_network'](seed,nO,nA,nB,params)
        self.target_critic = params['critic_network'](seed,nO,nA,nB,params)
        print('critic_type',self.critic_type)
        if self.critic_type == 'q':
            self.critic_backward = self.qcritic_backward
            self.actor_backward = self.qactor_backward
            self.critique = self.qcritique
        else:
            self.critic_backward = self.reg_critic_backward
            self.actor_backward = self.reg_actor_backward
            self.critique = self.reg_critique

        self.actor_optimizer = optim.Adam(self.local_actor.parameters(), lr=1e-8,weight_decay=params['L2'])
        self.critic_optimizer = optim.Adam(self.local_critic.parameters(), lr=1e-5)
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

    def qcritic_backward(self,critic_inputs:dict):
        """Computes critic grad update. Optionally computes betsize grad update in unison"""
        critic_output = self.local_critic(critic_inputs['observations'])
        value_mask = self.return_value_mask(critic_inputs['actions'])
        scaled_rewards = self.scale_rewards(critic_inputs['rewards'])
        # pre_value = critic_output['value'][value_mask]
        # print('critic_inputs',critic_inputs['observations'])
        critic_loss = F.smooth_l1_loss(scaled_rewards.view(value_mask.size(0)),critic_output['value'][value_mask])
        # print('scaled_rewards',scaled_rewards)
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
                # print('betsize_loss',betsize_loss)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.local_critic.parameters(), self.gradient_clip)
        self.critic_optimizer.step()
        Agent.soft_update(self.local_critic,self.target_critic,self.tau)
        # Assert learning
        # critic_output = self.local_critic(critic_inputs['observations'])
        # value_mask = self.return_value_mask(critic_inputs['actions'])
        # scaled_rewards = (critic_inputs['rewards']/self.max_reward).squeeze(1)
        # post_value = critic_output['value'][value_mask]
        # print('post update values')
            
    def qactor_backward(self,actor_inputs:dict,critic_inputs:dict):
        """
        critic_inputs: rewards,obs,actions,betsizes,betsize_prob,betsize_probs
        critic_outputs: value,betsize (Q values over action categories and betsize categories)
        actor_inputs: actions,action_prob,action_probs,game_states
        actor_outputs: action,action_prob,action_probs,betsize,betsize_prob,betsize_probs
        """
        critic_outputs = self.target_critic(critic_inputs['observations'])
        value_mask = self.return_value_mask(actor_inputs['actions'])
        values = critic_outputs['value']
        expected_value = (actor_inputs['action_probs'].view(-1) * values.view(-1)).detach().sum(-1)
        advantages = values[value_mask] - expected_value
        policy_loss = (-actor_inputs['action_prob'].view(-1) * advantages).sum() / 10
        pre_probs = actor_inputs['action_probs']
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
        actor_out = self.local_actor(actor_inputs['game_states'],actor_inputs['action_masks'],actor_inputs['betsize_masks'])
        post_probs = actor_out['action_probs']
        print('post')

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
        
"""
DQN with Priority Replay, DDQN, and Dueling DQN.
"""
class Priority_DQN(object):
    def __init__(self,state_space,obs_space,action_space,seed,params):
        self.action_space = action_space
        self.state_space = state_space
        self.seed = random.seed(seed)
        self.batch_size = params['BATCH_SIZE']
        self.buffer_size = params['BUFFER_SIZE']
        self.min_buffer_size = params['MIN_BUFFER_SIZE']
        self.learning_rate = params['LEARNING_RATE']
        self.update_every = params['UPDATE_EVERY']
        self.GAMMA = params['GAMMA']
        self.alpha = params['ALPHA']
        self.tau = params['TAU']
        self.clip_norm = params['CLIP_NORM']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.index = 0
        self.action_probs = V(torch.tensor([0.]),requires_grad=True)
        self.qnetwork_local = Dueling_QNetwork(seed,state_space,action_space).to(device)
        self.qnetwork_target = Dueling_QNetwork(seed,state_space,action_space).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(),lr=self.learning_rate)
        # Initialize replaybuffer
        self.memory = PriorityReplayBuffer(self.action_space,self.buffer_size,self.batch_size,self.seed,self.alpha)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
    def __call__(self,state,mask,eps=5.):
        state = state.float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        if random.random() > eps:
            masked_actions = (action_values + torch.min(action_values)) * mask
            action = masked_actions.max(1)[1]
            return action,self.action_probs
        else:
            eps *= 0.98
            prob_mask = mask / mask.sum()
            action = torch.tensor(np.random.choice(np.arange(self.action_space),p=prob_mask.numpy()))
            return action,self.action_probs
        
    def learn(self,player_data):
        positions = player_data.keys()
        for position in positions:
            # action_probs = player_data[position]['action_probs']
            # observations = player_data[position]['observations']
            actions = player_data[position]['actions']
            game_states = player_data[position]['game_states']
            rewards = player_data[position]['rewards']
        self.store_trajectories(game_states,actions,rewards)
        
    def store_trajectories(self,states,actions,rewards):
        next_states = states[1:]
        states = states[:-1]
        for i in range(len(states)):
            state = states[i]
            action = actions[i]
            reward = rewards[i]
            next_state = next_states[i]
            done = 0
            # Calculate TD error
            # Target
            current_network_action = self.qnetwork_local(next_state).max(1)[1]
            # initial state comes in as (1,4), squeeze to get (4)
            target = reward + self.GAMMA*(self.qnetwork_target(next_state).squeeze(0)[current_network_action])
            # Local. same rational for squeezing
            local = self.qnetwork_local(state).squeeze(0)[action]
            TD_error = reward + target - local
            # Save the experience
            self.memory.add(state,action,reward,next_state,done,TD_error,self.index)
            
            # learn from the experience
            self.t_step = (self.t_step + 1) % self.update_every
            if self.t_step == 0:
                if len(self.memory) > self.min_buffer_size:
                    experiences,indicies,weights = self.memory.sample(self.index)
                    self.backward(experiences,indicies,weights)
            self.index += 1

    def backward(self,experiences,indicies,weights):
        states,actions,rewards,next_states,dones = experiences
        # Local max action
        local_next_state_actions = self.qnetwork_local(next_states).max(1)[1].unsqueeze(1)
        # Target
        target_values = self.qnetwork_target(next_states).detach()
        max_target = target_values.gather(1,local_next_state_actions)
        max_target *= (1-dones) 
        targets = rewards + (self.GAMMA*max_target)
#         targets = rewards + self.GAMMA*(target_values.gather(1,local_next_state_actions))
        # Local
        local = self.qnetwork_local(states).gather(1,actions)
        TD_error = local - targets
        if TD_error.is_cuda:
            loss = ((torch.tensor(weights).cuda() * TD_error)**2*0.5).mean()
        else:
            loss = ((torch.tensor(weights) * TD_error)**2*0.5).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(),self.clip_norm)
        self.optimizer.step()
        # Update the priorities
        TD_errors = np.abs(TD_error.squeeze(1).detach().cpu().numpy())
        self.memory.sum_tree.update_priorities(TD_errors,indicies)
        self.update_target()
        
    # Polyak averaging  
    def update_target(self):
        for local_param,target_param in zip(self.qnetwork_local.parameters(),self.qnetwork_target.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1-self.tau)*target_param.data)