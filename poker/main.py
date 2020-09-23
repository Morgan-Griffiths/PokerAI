
import time
import torch.multiprocessing as mp
import torch
import os

from train import train,train_dual,generate_trajectories,dual_learning_update,combined_learning_update
from poker_env.config import Config
import poker_env.datatypes as pdt
from poker_env.env import Poker
from db import MongoDB
from models.network_config import NetworkConfig,CriticType
from models.networks import OmahaActor,OmahaQCritic,CombinedNet
from models.model_utils import update_weights,hard_update
from utils.utils import unpack_shared_dict

from torch import optim

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=
        """
        Train RL algorithms in a poker environment
        """)

    parser.add_argument('--network','-n',
                        dest='network_type',
                        default='combined',
                        metavar="['combined','dual']",
                        type=str,
                        help='whether to split the actor critic into two separate networks or not')

    args = parser.parse_args()

    print("Number of processors: ", mp.cpu_count())
    print(f'Number of GPUs: {torch.cuda.device_count()}')
    tic = time.time()

    config = Config()
    game_object = pdt.Globals.GameTypeDict[pdt.GameTypes.OMAHAHI]

    env_params = {
        'game':pdt.GameTypes.OMAHAHI,
        'betsizes': game_object.rule_params['betsizes'],
        'bet_type': game_object.rule_params['bettype'],
        'n_players': 2,
        'pot':1,
        'stacksize': game_object.state_params['stacksize'],
        'cards_per_player': game_object.state_params['cards_per_player'],
        'starting_street': game_object.starting_street,
        'global_mapping':config.global_mapping,
        'state_mapping':config.state_mapping,
        'obs_mapping':config.obs_mapping,
        'shuffle':True
    }
    print(f'Environment Parameters: Starting street: {env_params["starting_street"]},\
        Stacksize: {env_params["stacksize"]},\
        Pot: {env_params["pot"]},\
        Bettype: {env_params["bet_type"]},\
        Betsizes: {env_params["betsizes"]}')
    env = Poker(env_params)

    nS = env.state_space
    nA = env.action_space
    nB = env.betsize_space
    seed = 1235
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu1 = 'cuda:0'
    gpu2 = 'cuda:1'

    network_params = {
        'game':pdt.GameTypes.OMAHAHI,
        'maxlen':config.maxlen,
        'state_mapping':config.state_mapping,
        'embedding_size':128,
        'transformer_in':1280,
        'transformer_out':128,
        'device':device,
        'frozen_layer_path':'../hand_recognition/checkpoints/regression/PartialHandRegression'
    }
    training_params = {
        'training_epochs':50,
        'epochs':10,
        'training_round':0,
        'game':'Omaha',
        'id':0
    }
    learning_params = {
        'training_round':0,
        'gradient_clip':config.agent_params['CLIP_NORM'],
        'path': os.path.join(os.getcwd(),'checkpoints'),
        'learning_rounds':1,
        'device':device,
        'gpu1':gpu1,
        'gpu2':gpu2,
        'min_reward':-env_params['stacksize'],
        'max_reward':env_params['pot']+env_params['stacksize']
    }
    path = learning_params['path']
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.mkdir(directory)
    # Clean mongo
    mongo = MongoDB()
    mongo.clean_db()
    mongo.close()
    # Set processes
    mp.set_start_method('spawn')
    num_processes = min(mp.cpu_count(),2)
    if args.network_type == 'combined':
        alphaPoker = CombinedNet(seed,nS,nA,nB,network_params).to(device)
        alphaPoker_optimizer = optim.Adam(alphaPoker.parameters(), lr=config.agent_params['critic_lr'])
        learning_params['model_optimizer'] = alphaPoker_optimizer
        alphaPoker.share_memory()#.to(device)
        processes = []
        # for debugging
        # generate_trajectories(env,alphaPoker,training_params,id=0)
        # alphaPoker,learning_params = combined_learning_update(alphaPoker,learning_params)
        # train(env,alphaPoker,training_params,learning_params,id=0)
        for id in range(num_processes): # No. of processes
            p = mp.Process(target=train, args=(env,alphaPoker,training_params,learning_params,id))
            p.start()
            processes.append(p)
        for p in processes: 
            p.join()
        # save weights
        torch.save(alphaPoker.state_dict(), os.path.join(path,'RL_combined'))
    else:
        actor = OmahaActor(seed,nS,nA,nB,network_params)
        critic = OmahaQCritic(seed,nS,nA,nB,network_params)
        actor_optimizer = optim.Adam(actor.parameters(), lr=config.agent_params['actor_lr'],weight_decay=config.agent_params['L2'])
        critic_optimizer = optim.Adam(critic.parameters(), lr=config.agent_params['critic_lr'])
        learning_params['actor_optimizer'] = actor_optimizer
        learning_params['critic_optimizer'] = critic_optimizer
        # training loop
        actor.share_memory()#.to(device)
        critic.share_memory()#.to(device)

        processes = []
        # for debugging
        # generate_trajectories(env,actor,training_params,id=0)
        # actor,critic,learning_params = dual_learning_update(actor,critic,learning_params)
        # train_dual(env,actor,critic,training_params,learning_params,id=0)
        for id in range(num_processes): # No. of processes
            p = mp.Process(target=train_dual, args=(env,actor,critic,training_params,learning_params,id))
            p.start()
            processes.append(p)
        for p in processes: 
            p.join()
        # save weights
        torch.save(actor.state_dict(), os.path.join(path,'RL_actor'))
        torch.save(critic.state_dict(), os.path.join(path,'RL_critic'))
