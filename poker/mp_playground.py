
import time
import torch.multiprocessing as mp
import torch
import os
import datetime
from torch.optim.lr_scheduler import MultiStepLR,StepLR

from train import train_combined,train_dual,train_batch,generate_trajectories,dual_learning_update,combined_learning_update,train_test
from poker_env.config import Config
import poker_env.datatypes as pdt
from poker_env.env import Poker
from db import MongoDB
from models.network_config import NetworkConfig,CriticType
from models.networks import OmahaActor,OmahaQCritic,OmahaObsQCritic,CombinedNet,BetAgent
from models.model_utils import copy_weights,hard_update,expand_conv2d,load_weights
from utils.utils import unpack_shared_dict,clean_folder,return_latest_baseline_path,return_next_baseline_path,return_latest_training_model_path
from tournament import tournament,print_stats,eval_latest

from torch import optim


def example(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # create default process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # create local model
    model = nn.Linear(10, 10).to(rank)
    # construct DDP model
    ddp_model = DDP(model, device_ids=[rank])
    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    # forward pass
    outputs = ddp_model(torch.randn(20, 10).to(rank))
    labels = torch.randn(20, 10).to(rank)
    # backward pass
    loss_fn(outputs, labels).backward()
    # update parameters
    optimizer.step()
    cleanup()

def main():
    world_size = 2
    mp.spawn(example,
        args=(world_size,),
        nprocs=world_size,
        join=True)



if __name__ == '__main__':
    EPOCHS = 1

    num_gpus = torch.cuda.device_count()
    print("Number of processors: ", mp.cpu_count())
    print(f'Number of GPUs: {num_gpus}')
    tic = time.time()

    config = Config()
    game_object = pdt.Globals.GameTypeDict[pdt.GameTypes.OMAHAHI]

    env_params = {
        'game':pdt.GameTypes.OMAHAHI,
        'betsizes': game_object.rule_params['betsizes'],
        'bet_type': game_object.rule_params['bettype'],
        'n_players': 2,
        'pot': game_object.state_params['pot'],
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
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    network_params                                = config.network_params
    network_params['device'] = device
    network_params['seed'] = seed
    network_params['nS'] = nS
    network_params['nA'] = nA
    network_params['nB'] = nB
    training_params = {
        'lr_steps':1,
        'training_epochs':EPOCHS,
        'generate_epochs':1,
        'training_round':0,
        'game':'Omaha',
        'id':0,
        'save_every':max(EPOCHS // 4,1),
        'save_dir':os.path.join(os.getcwd(),'checkpoints/training_run'),
        'actor_path':config.agent_params['actor_path'],
        'critic_path':config.agent_params['critic_path'],
        'baseline_path':config.baseline_path
    }
    learning_params = {
        'training_round':0,
        'gradient_clip':config.agent_params['CLIP_NORM'],
        'learning_rounds':1,
        'device':device,
        'min_reward':-env_params['stacksize'],
        'max_reward':env_params['pot']+env_params['stacksize']
    }
    validation_params = {
        'actor_path':config.agent_params['actor_path'],
        'epochs':1,
        'koth':False
    }
    num_processes = min(1,num_gpus)
    main()
    # mp.spawn(train_test,
    # args=(env,training_params,learning_params,network_params,validation_params,),
    # nprocs=num_processes,
    # join=True)