import torch
from torch.autograd import Variable as V
import numpy as np
import os
from poker.multistreet_env import MSPoker
from agents.agent import FullAgent
from poker.config import Config
import poker.datatypes as pdt
from models.network_config import NetworkConfig
from models.networks import NetworkFunctions

{0:'check',1:'fold',2:'call',3:'bet',4:'raise',5:'unopened'}

# class HumanPlayer(object):
#     def __init__(self):
#         self.helper_functions = NetworkFunctions(self.nA,self.nB)
        
#     def get_input(self):
#         action_category,betsize_category = self.helper_functions.unwrap_action(action,state[:,-1,self.mapping['state']['previous_action']])
        

class PlayEnv(object):
    def __init__(self,nA,nB,betsizes,env,agent,training_params,player_position):
        self.nA = nA
        self.nB = nB
        self.nC = nA - 2 + nB # Flat action output over action categories and betsizes
        self.BETSIZE_ACTION_DICT = {
            0:'check',
            1:'fold',
            2:'call'
        }
        for i,betsize in enumerate(betsizes):
            self.BETSIZE_ACTION_DICT[i+3] = f'Bet {round(float(betsize),2)} of pot'

        self.env = env
        self.agent = agent
        self.training_params = training_params
        self.player_position = player_position
        self.helper_functions = NetworkFunctions(nA,nB)
        self.fake_prob = V(torch.tensor([0.]),requires_grad=True)

        self.get_player_input = self.get_player_input_betsize
        self.fake_probs = V(torch.Tensor(self.nC).fill_(0.),requires_grad=True) 

    def get_player_input_betsize(self,*args):
        action_mask,betsize_mask = args
        mask = torch.cat([action_mask[:-2],betsize_mask])
        improper_input = True
        player_input = input(f'Enter one of the following numbers {torch.arange(len(mask)).numpy()[mask.numpy().astype(bool)]}, {[self.BETSIZE_ACTION_DICT[action] for action in torch.arange(len(mask)).numpy()[mask.numpy().astype(bool)]]}')
        while improper_input == True:
            while player_input.isdigit() == False:
                print('Invalid input, please enter a number or exit with ^C')
                player_input = input(f'Enter one of the following numbers {torch.arange(len(mask)).numpy()[mask.numpy().astype(bool)]}, {[self.BETSIZE_ACTION_DICT[action] for action in torch.arange(len(mask)).numpy()[mask.numpy().astype(bool)]]}')
            if int(player_input) not in torch.arange(len(mask)).numpy()[mask.numpy().astype(bool)]:
                print('Invalid action choice, please choose one of the following')
                player_input = input(f'Enter one of the following numbers {torch.arange(len(mask)).numpy()[mask.numpy().astype(bool)]}, {[pdt.Globals.ACTION_DICT[action] for action in torch.arange(len(mask)).numpy()[mask.numpy().astype(bool)]]}')
            else:
                improper_input = False
        action = torch.tensor(int(player_input)).unsqueeze(0)
        return action
        
    def play(self):
        training_data = self.training_params['training_data']
        for e in range(self.training_params['epochs']):
            print(f'Hand {e}')
            state,obs,done,mask,betsize_mask = self.env.reset()
            print('Street',env.street)
            print(f'state {state[0,0]},done {done}')
            while not done:
                print(f'current_player {self.env.current_player}')
                board = [[card.rank,card.suit] for card in self.env.board]
                hand = [[card.rank,card.suit] for card in self.env.players.current_hand]
                print(f'Current board {[[pdt.Globals.POKER_RANK_DICT[card[0]],pdt.Globals.POKER_SUIT_DICT[card[1]]] for card in board]}')
                print(f'Current Hand {[[pdt.Globals.POKER_RANK_DICT[card[0]],pdt.Globals.POKER_SUIT_DICT[card[1]]] for card in hand]}')
                if self.env.current_player == self.player_position:
                    action = self.get_player_input(mask,betsize_mask)
                    last_action = state[-1,-1,self.env.db_mapping['state']['previous_action']]
                    print('last_action',last_action)
                    action_category,betsize_category = self.helper_functions.unwrap_action(action,last_action)
                    print('action_category,betsize_category',action_category,betsize_category)
                    outputs = {
                        'action':action,
                        'action_category':action_category,
                        'betsize':betsize_category
                        }
                    outputs['action_prob']= self.fake_prob
                    outputs['action_probs']= self.fake_probs
                else:
                    outputs = agent(state,mask,betsize_mask)
                    print(f'Action Probabilities {outputs["action_probs"]}')
                print(f'Action {outputs["action"]}')
                state,obs,done,mask,betsize_mask = env.step(outputs)
                print(f'state {state[0,env.game_turn.value]},done {done}')
            ml_inputs = env.ml_inputs()
            if player_position in ml_inputs:
                player_inputs = ml_inputs[player_position]
                print(f"Game state {player_inputs['game_states']}, Actions {player_inputs['actions']}, Rewards {player_inputs['rewards']}")
        return training_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=
        """
        Run RL experiments with varying levels of complexity in poker.\n\n
        Modify the traiing params via the config file.\n
        """)
    parser.add_argument('--game',
                        default=pdt.GameTypes.HOLDEM,
                        metavar=f"[{pdt.GameTypes.OMAHAHI},{pdt.GameTypes.HOLDEM}]",
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
    # config.params['game'] = args.game

    game_object = pdt.Globals.GameTypeDict[args.game]
    player_position = pdt.Globals.POSITION_DICT[args.position]
    
    params = {'game':args.game}
    params['state_params'] = game_object.state_params
    params['maxlen'] = config.maxlen
    params['rule_params'] = game_object.rule_params
    # params['state_params'] = pdt.Globals.GameTypeDict[args.game].state_params
    # params['rule_params'] = pdt.Globals.GameTypeDict[args.game].rule_params
    training_params = config.training_params
    agent_params = config.agent_params
    env_networks = NetworkConfig.EnvModels[args.game]
    agent_params['network'] = env_networks['actor']
    agent_params['actor_network'] = env_networks['actor']
    agent_params['critic_network'] = env_networks['critic']['q']
    agent_params['mapping'] = params['rule_params']['mapping']
    agent_params['max_reward'] = params['state_params']['stacksize'] + params['state_params']['pot']
    agent_params['min_reward'] = params['state_params']['stacksize']
    agent_params['epochs'] = 0
    params['rule_params']['network_output'] = 'flat'
    params['rule_params']['betsizes'] = pdt.Globals.BETSIZE_DICT[2]
    agent_params['network_output'] = 'flat'
    agent_params['embedding_size'] = 32
    agent_params['maxlen'] = config.maxlen
    agent_params['game'] = args.game
    params['starting_street'] = game_object.starting_street

    training_data = {}
    for position in pdt.Globals.PLAYERS_POSITIONS_DICT[params['state_params']['n_players']]:
        training_data[position] = []
    training_params['training_data'] = training_data
    training_params['agent_name'] = f'{args.game}_baseline'

    env = MSPoker(params)

    nS = env.state_space
    nO = env.observation_space
    nA = env.action_space
    nB = env.rules.num_betsizes
    print(f'Environment: State Space {nS}, Obs Space {nO}, Action Space {nA}')
    seed = 154

    agent = FullAgent(nS,nO,nA,nB,seed,agent_params)
    agent.load_weights(os.path.join(training_params['save_dir'],'Holdem'))

    play_env = PlayEnv(nA,nB,params['rule_params']['betsizes'],env,agent,training_params,player_position)
    action_data = play_env.play()