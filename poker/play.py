import torch
from torch.autograd import Variable as V
import numpy as np
from config import Config
from environment import Poker
from models.utils import return_agent

action_dict = {0:'check',1:'bet',2:'call',3:'fold',4:'raise',5:'unopened'}
card_dict = {0:'?',1:'Q',2:'K',3:'A'}
mask_dict = {
        5:torch.Tensor([1,1,0,1]),
        0:torch.Tensor([0,0,0,0]),
        1:torch.Tensor([0,0,1,1]),
        2:torch.Tensor([0,0,0,0])
        }

def return_mask(state,action_index):
    return mask_dict[state[0,action_index].long().item()]

def get_player_input(mask):
    player_input = input(f'Enter one of the following numbers {torch.arange(4).numpy()[mask.numpy().astype(bool)]}, {[action_dict[action] for action in torch.arange(4).numpy()[mask.numpy().astype(bool)]]}')
    while player_input.isdigit() == False:
        print('Invalid input, please enter a number or exit with ^C')
        player_input = input(f'Enter one of the following numbers {torch.arange(4).numpy()[mask.numpy().astype(bool)]}, {[action_dict[action] for action in torch.arange(4).numpy()[mask.numpy().astype(bool)]]}')
    action = torch.tensor(int(player_input))
    return action
    
def play(env,agent,player_position,training_params):
    fake_probs = V(torch.tensor([0.]),requires_grad=True)   
    training_data = training_params['training_data']
    for e in range(training_params['epochs']):
        print(f'Hand {e}')
        state,obs,done = env.reset()
        mask = return_mask(state,training_params['action_index'])
        print(f'state {state},done {done}')
        while not done:
            print(f'current_player {env.current_player}, Current Hand {card_dict[env.players.current_hand.rank]}')
            if env.current_player == player_position:
                action = get_player_input(mask)
                log_probs = fake_probs
            else:
                action,log_probs = agent(state,mask)
                print(f'log_probs {log_probs}')
            print(f'Action {action}')
            state,obs,done = env.step(action,log_probs)
            print(f'state {state},done {done}')
            if state[-1] != -1:
                mask = return_mask(state,training_params['action_index'])
        ml_inputs = env.ml_inputs()
        player_inputs = ml_inputs[player_position]
        print(f"Game state {player_inputs['game_states']}, Actions {player_inputs['actions']}, Rewards {player_inputs['rewards']}")
        # agent.learn(ml_inputs)
        # for position in ml_inputs.keys():
        #     training_data[position].append(ml_inputs[position])
    return training_data


if __name__ == "__main__":
    config = Config()
    # config.agent = args.agent
    # config.params['game'] = args.env

    params = config.params
    agent_params = config.agent_params
    position_dict = config.position_dict

    training_data = {}
    for position in position_dict[params['state_params']['n_players']]:
        training_data[position] = []
        
    training_params = {
        'epochs':2500,
        'action_index':1,
        'training_data':training_data,
        'training_round':0
    }

    env = Poker(params)

    nS = env.state_space[0]
    nO = env.observation_space[0]
    nA = env.action_space
    print(f'Environment: State Space {nS}, Obs Space {nO}, Action Space {nA}')
    seed = 154
    player_position = 'SB'

    agent = return_agent(config.agent,nS,nO,nA,seed,agent_params)
    action_data = play(env,agent,player_position,training_params)