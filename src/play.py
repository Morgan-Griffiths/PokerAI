import torch
from torch.autograd import Variable as V
import numpy as np
import os
from poker_env import Poker
from agents.agent import return_agent
from poker.config import Config
import poker.datatypes as pdt
from models.network_config import NetworkConfig

def get_player_input(mask):
    player_input = input(f'Enter one of the following numbers {torch.arange(len(mask)).numpy()[mask.numpy().astype(bool)]}, {[pdt.Globals.ACTION_DICT[action] for action in torch.arange(len(mask)).numpy()[mask.numpy().astype(bool)]]}')
    while player_input.isdigit() == False:
        print('Invalid input, please enter a number or exit with ^C')
        player_input = input(f'Enter one of the following numbers {torch.arange(len(mask)).numpy()[mask.numpy().astype(bool)]}, {[pdt.Globals.ACTION_DICT[action] for action in torch.arange(len(mask)).numpy()[mask.numpy().astype(bool)]]}')
    action = torch.tensor(int(player_input)).unsqueeze(0)
    return action
    
def play(env,agent,player_position,training_params):
    fake_probs = V(torch.tensor([0.]),requires_grad=True)   
    training_data = training_params['training_data']
    for e in range(training_params['epochs']):
        print(f'Hand {e}')
        state,obs,done = env.reset()
        mask = env.action_mask(state)
        print(f'state {state},done {done}')
        while not done:
            print(f'current_player {env.current_player}')
            if len(env.players.current_hand) == 1:
                print(f'Current Hand {pdt.Globals.KUHN_CARD_DICT[env.players.current_hand.rank]}')
            else:
                board = [[card.rank,card.suit] for card in env.board]
                hand = [[card.rank,card.suit] for card in env.players.current_hand]
                print(f'Current board {[[pdt.Globals.HOLDEM_RANK_DICT[card[0]],pdt.Globals.HOLDEM_SUIT_DICT[card[1]]] for card in board]}')
                print(f'Current Hand {[[pdt.Globals.HOLDEM_RANK_DICT[card[0]],pdt.Globals.HOLDEM_SUIT_DICT[card[1]]] for card in hand]}')

            if env.current_player == player_position:
                action = get_player_input(mask)
                log_probs = fake_probs
            else:
                action,log_probs = agent(state,mask)
                print(f'log_probs {log_probs}')
            print(f'Action {action}')
            state,obs,done = env.step(action,log_probs)
            print(f'state {state},done {done}')
            mask = env.action_mask(state)
        ml_inputs = env.ml_inputs()
        if player_position in ml_inputs:
            player_inputs = ml_inputs[player_position]
            print(f"Game state {player_inputs['game_states']}, Actions {player_inputs['actions']}, Rewards {player_inputs['rewards']}")
        # agent.learn(ml_inputs)
        # for position in ml_inputs.keys():
        #     training_data[position].append(ml_inputs[position])
    return training_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=
        """
        Run RL experiments with varying levels of complexity in poker.\n\n
        Modify the traiing params via the config file.\n
        """)
    parser.add_argument('--env',
                        default=pdt.GameTypes.KUHN,
                        metavar=f"[{pdt.GameTypes.KUHN},{pdt.GameTypes.COMPLEXKUHN},{pdt.GameTypes.HOLDEM}]",
                        help='Picks which type of poker env to play')
    parser.add_argument('-p','--pos',
                        dest='position',
                        default=pdt.Positions.SB,
                        metavar=f"[{pdt.Positions.SB},{pdt.Positions.BB}]",
                        help='Picks which position to play')

    args = parser.parse_args()

    print(args)

    config = Config()
    # config.agent = args.agent
    # config.params['game'] = args.env


    config = Config()
    player_position = pdt.Globals.POSITION_DICT[args.position]
    
    params = {'game':args.env}
    params['state_params'] = pdt.Globals.GameTypeDict[args.env].state_params
    params['rule_params'] = pdt.Globals.GameTypeDict[args.env].rule_params
    training_params = config.training_params
    agent_params = config.agent_params
    agent_params['network'] = NetworkConfig.EnvModels[args.env]
    agent_params['mapping'] = params['rule_params']['mapping']

    training_data = {}
    for position in pdt.Globals.PLAYERS_POSITIONS_DICT[params['state_params']['n_players']]:
        training_data[position] = []
    training_params['training_data'] = training_data
    training_params['agent_name'] = f'{args.env}_baseline'

    env = Poker(params)

    nS = env.state_space
    nO = env.observation_space
    nA = env.action_space
    print(f'Environment: State Space {nS}, Obs Space {nO}, Action Space {nA}')
    seed = 154

    agent = return_agent(config.agent,nS,nO,nA,seed,agent_params)
    agent.load_weights(os.path.join(training_params['save_dir'],training_params['agent_name']))
    action_data = play(env,agent,player_position,training_params)