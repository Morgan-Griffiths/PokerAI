from train import train
from poker.config import Config
from poker_env import Poker
from agents.agent import Agent,Priority_DQN,return_agent
from db import MongoDB
import poker.datatypes as dt

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=
        """
        Run RL experiments with varying levels of complexity in poker.\n\n
        Modify the traiing params via the config file.\n
        """)

    parser.add_argument('--agent',
                        default='baseline',
                        metavar="['baseline','dqn']",
                        help='Which agent to train')
    parser.add_argument('--env',
                        default=dt.GameTypes.KUHN,
                        metavar=f"[{dt.GameTypes.KUHN},{dt.GameTypes.COMPLEXKUHN},{dt.GameTypes.HOLDEM}]",
                        help='Picks which type of poker env to train in')
    parser.add_argument('--clean',
                        default=False,
                        metavar="boolean",
                        help='cleans database')
    parser.add_argument('--epochs',
                        default=1000,
                        metavar="int",
                        help='Number of training epochs')

    args = parser.parse_args()

    print(f'args {args}')

    env_type = 'complex'

    config = Config(env_type)
    config.agent = args.agent
    config.params['game'] = args.env

    params = config.params
    params['rule_params'] = dt.Globals.GameTypeDict[args.env].rule_params
    agent_params = config.agent_params
    position_dict = config.position_dict

    training_data = {}
    for position in position_dict[params['state_params']['n_players']]:
        training_data[position] = []
        
    training_params = config.training_params
    training_params['epochs'] = int(args.epochs)
    training_params['training_data'] = training_data
    training_params['agent_name'] = args.agent
    # Change pot size to see how it effects betting and calling
    # params['state_params']['pot'] = 5

    env = Poker(params)

    nS = env.state_space
    nO = env.observation_space
    nA = env.action_space
    print(f'Environment: State Space {nS}, Obs Space {nO}, Action Space {nA}')
    seed = 154

    agent = return_agent(config.agent,nS,nO,nA,seed,agent_params)
    action_data = train(env,agent,training_params)

    mongo = MongoDB()
    if args.clean:
        print('Cleaning db')
        mongo.clean_db()
    mongo.store_data(action_data,env.db_mapping,training_params['training_round'])