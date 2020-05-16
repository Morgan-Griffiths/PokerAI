from train import train
from poker.config import Config
from multistreet_env import MSPoker
from poker_env import Poker
from agents.agent import Agent,Priority_DQN,return_agent
from db import MongoDB
import poker.datatypes as pdt
from models.network_config import NetworkConfig,CriticType
import time
import multiprocessing as mp

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=
        """
        Run RL experiments with varying levels of complexity in poker.\n\n
        Modify the traiing params via the config file.\n
        """)

    parser.add_argument('--agent',
                        default='actor_critic',
                        metavar="['actor','actor_critic']",
                        type=str,
                        help='Which agent to train')
    parser.add_argument('--env',
                        default=pdt.GameTypes.HOLDEM,
                        type=str,
                        metavar=f"[{pdt.GameTypes.KUHN},{pdt.GameTypes.COMPLEXKUHN},{pdt.GameTypes.BETSIZEKUHN},{pdt.GameTypes.HOLDEM}]",
                        help='Picks which type of poker env to train in')
    parser.add_argument('--no-clean',
                        default=True,
                        dest='clean',
                        action='store_false',
                        help='Cleans database')
    parser.add_argument('--no-store',
                        default=True,
                        dest='store',
                        action='store_false',
                        help='Stores training data in database')
    parser.add_argument('-e','--epochs',
                        default=200,
                        type=int,
                        help='Number of training epochs')
    parser.add_argument('--critic',
                        default='q',
                        type=str,
                        metavar="['q','reg']",
                        help='Critic output types [nA,1]')
    parser.add_argument('--betsize',
                        default=2,
                        type=int,
                        metavar="[1-5 or 11]",
                        help='Number of betsizes the agent can make')
    parser.add_argument('--N-output',
                        default='flat',
                        dest='network_output',
                        type=str,
                        metavar="['flat','tiered']",
                        help='Network output types. Controls whether betsize is combined with actions or not')
    parser.add_argument('--padding',
                        default=True,
                        type=bool,
                        help='To pad the network inputs')
    parser.add_argument('--maxlen',
                        default=20,
                        dest='padding_maxlen',
                        type=int,
                        help='Size of padding')

    args = parser.parse_args()

    print("Number of processors: ", mp.cpu_count())
    print(f'args {args}')
    tic = time.time()
    
    game_object = pdt.Globals.GameTypeDict[args.env]
    config = Config()
    config.agent = args.agent
    params = {'game':args.env}
    params['state_params'] = game_object.state_params
    params['rule_params'] = game_object.rule_params
    params['rule_params']['network_output'] = args.network_output
    params['rule_params']['betsizes'] = pdt.Globals.BETSIZE_DICT[args.betsize]
    params['rule_params']['maxturns'] = args.padding_maxlen
    params['rule_params']['padding'] = args.padding
    params['starting_street'] = game_object.starting_street
    agent_params = config.agent_params

    env_networks = NetworkConfig.EnvModels[args.env]
    agent_params['network'] = env_networks['actor']
    agent_params['actor_network'] = env_networks['actor']
    agent_params['critic_network'] = env_networks['critic'][args.critic]
    agent_params['mapping'] = params['rule_params']['mapping']
    agent_params['max_reward'] = params['state_params']['stacksize'] + params['state_params']['pot']
    agent_params['min_reward'] = params['state_params']['stacksize']
    agent_params['epochs'] = int(args.epochs)
    agent_params['network_output'] = args.network_output
    agent_params['embedding_size'] = 32
    agent_params['max_length'] = args.padding_maxlen

    print(f'Training the following networks {agent_params["critic_network"].__name__},{agent_params["actor_network"].__name__}')

    training_data = {}
    for position in pdt.Globals.PLAYERS_POSITIONS_DICT[params['state_params']['n_players']]:
        training_data[position] = []
    training_data['action_records'] = []
        
    training_params = config.training_params
    training_params['epochs'] = int(args.epochs)
    training_params['training_data'] = training_data
    training_params['agent_name'] = f'{args.env}_baseline'
    training_params['agent_type'] = args.agent
    training_params['critic'] = args.critic

    if args.env == pdt.GameTypes.HOLDEM or args.env == pdt.GameTypes.OMAHAHI:
        env = MSPoker(params)
    else:
        env = Poker(params)

    nS = env.state_space
    nO = env.observation_space
    nA = env.action_space
    nB = env.betsize_space
    nC = nA - 2 + nB
    print(f'Environment: State Space {nS}, Obs Space {nO}, Action Space {nA}, Betsize Space {nB}, Flat Action Space {nC}')
    seed = 154

    agent = return_agent(args.agent,nS,nO,nA,nB,seed,agent_params)

    action_data = train(env,agent,training_params)
    # print(action_data)
    if args.store:
        print('\nStoring training data')
        mongo = MongoDB()
        if args.clean:
            print('Cleaning db')
            mongo.clean_db()
        mongo.store_data(action_data,env.db_mapping,training_params['training_round'],env.game)

    toc = time.time()
    print(f'\nExecution took {(toc-tic)/60} minutes')