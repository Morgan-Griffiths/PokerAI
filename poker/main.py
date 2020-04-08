from train import train
from config import Config
from environment import Poker
from agents.agent import Agent,Priority_DQN
from db import MongoDB
from models.utils import return_agent

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
                        default='kuhn',
                        metavar="['kuhn']",
                        help='Picks which type of poker env to train in')
    parser.add_argument('--clean',
                        default=False,
                        metavar="boolean",
                        help='cleans database')

    args = parser.parse_args()

    print(f'args {args}')

    config = Config()
    config.agent = args.agent
    config.params['game'] = args.env

    params = config.params
    agent_params = config.agent_params
    position_dict = config.position_dict

    training_data = {}
    for position in position_dict[params['state_params']['n_players']]:
        training_data[position] = []
        
    training_params = {
        'epochs':1000,
        'action_index':1,
        'training_data':training_data,
        'training_round':0
    }

    env = Poker(params)

    nS = env.state_space
    nO = env.observation_space
    nA = env.action_space
    print(f'Environment: State Space {nS}, Obs Space {nO}, Action Space {nA}')
    seed = 154

    agent = return_agent(config.agent,nS,nO,nA,seed,agent_params)
    action_data = train(env,agent,training_params)
    mapping = {'state':{'previous_action':1,'hand':0},
        'observation':{'previous_action':1,'hand':0},
        }

    mongo = MongoDB()
    if args.clean:
        print('Cleaning db')
        mongo.clean_db()
    mongo.store_data(action_data,mapping,training_params['training_round'])