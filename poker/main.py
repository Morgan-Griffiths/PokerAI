
import time
import torch.multiprocessing as mp
import torch
import os
from torch.optim.lr_scheduler import MultiStepLR,StepLR

from train import train,train_dual,train_batch,generate_trajectories,dual_learning_update,combined_learning_update
from poker_env.config import Config
import poker_env.datatypes as pdt
from poker_env.env import Poker
from db import MongoDB
from models.network_config import NetworkConfig,CriticType
from models.networks import OmahaActor,OmahaQCritic,OmahaObsQCritic,CombinedNet
from models.model_utils import copy_weights,hard_update,expand_conv2d
from models.model_layers import ProcessHandBoard,PreProcessLayer
from utils.utils import unpack_shared_dict,clean_folder

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
                        default='dual',
                        metavar="['combined','dual']",
                        type=str,
                        help='whether to split the actor critic into two separate networks or not')
    parser.add_argument('--epochs','-e',
                        dest='epochs',
                        default=10,
                        type=int,
                        help='Number of training rounds')
    parser.add_argument('--generate','-g',
                        dest='generate',
                        default=5,
                        type=int,
                        help='Number of generated hands per epoch per thread')
    parser.add_argument('--learning','-l',
                        dest='learning',
                        default=1,
                        type=int,
                        help='Number of learning passes on the data')
    parser.add_argument('--steps','-s',
                        dest='steps',
                        default=3,
                        type=int,
                        help='Number of learning rate decays')
    parser.add_argument('--frozen',
                        dest='frozen',action='store_true',
                        help='Preload handboard recognizer weights')
    parser.add_argument('--no-frozen',
                        dest='frozen',action='store_false',
                        help='Do not preload handboard recognizer weights')
    parser.add_argument('--gpu',
                        dest='gpu',
                        default=0,
                        type=int,
                        help='Which gpu to use')
    parser.set_defaults(frozen=True)

    args = parser.parse_args()

    cuda_dict = {0:'cuda:0',1:'cuda:1'}

    print(args)
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
    device = torch.device(cuda_dict[args.gpu] if torch.cuda.is_available() else "cpu")
    gpu1 = 'cuda:0'
    gpu2 = 'cuda:1'
    network_params                                = config.network_params
    network_params['device']                      = device
    training_params = {
        'lr_steps':args.steps,
        'training_epochs':args.epochs,
        'generate_epochs':args.generate,
        'training_round':0,
        'game':'Omaha',
        'id':0,
        'save_every':max(args.epochs // 4,1),
        'save_dir':os.path.join(os.getcwd(),'checkpoints/training_run'),
        'actor_path':config.agent_params['actor_path'],
        'critic_path':config.agent_params['critic_path'],
    }
    learning_params = {
        'training_round':0,
        'gradient_clip':config.agent_params['CLIP_NORM'],
        'learning_rounds':args.learning,
        'device':device,
        'gpu1':gpu1,
        'gpu2':gpu2,
        'min_reward':-env_params['stacksize'],
        'max_reward':env_params['pot']+env_params['stacksize']
    }
    path = training_params['save_dir']
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.mkdir(directory)
    # Clean training_run folder
    # clean_folder(training_params['save_dir'])
    # Clean mongo
    mongo = MongoDB()
    mongo.clean_db()
    mongo.close()
    # Set processes
    mp.set_start_method('spawn')
    num_processes = min(mp.cpu_count(),6)
    print(f'Number of used processes {num_processes}')
    print(f'Training {args.network_type} model')
    if args.network_type == 'combined':
        alphaPoker = CombinedNet(seed,nS,nA,nB,network_params).to(device)
        if args.frozen:
            # Load pretrained hand recognizer
            copy_weights(alphaPoker,network_params['actor_hand_recognizer_path'])
        alphaPoker.summary
        alphaPoker_optimizer = optim.Adam(alphaPoker.parameters(), lr=config.agent_params['critic_lr'])
        lrscheduler = StepLR(alphaPoker_optimizer, step_size=1, gamma=0.1)
        learning_params['model_optimizer'] = alphaPoker_optimizer
        learning_params['lrscheduler'] = lrscheduler
        alphaPoker.share_memory()
        processes = []
        # for debugging
        # generate_trajectories(env,alphaPoker,training_params,id=0)
        # alphaPoker,learning_params = combined_learning_update(alphaPoker,learning_params)
        # train(env,alphaPoker,training_params,learning_params,id=0)
        for e in range(training_params['lr_steps']):
            for id in range(num_processes): # No. of processes
                p = mp.Process(target=train, args=(env,alphaPoker,training_params,learning_params,id))
                p.start()
                processes.append(p)
            for p in processes: 
                p.join()
            learning_params['lrscheduler'].step()
            training_params['training_round'] = (e+1) * training_params['training_epochs']
        # save weights
        torch.save(alphaPoker.state_dict(), os.path.join(path,'OmahaCombinedFinal'))
        print(f'Saved model weights to {os.path.join(path,"OmahaCombinedFinal")}')
    elif args.network_type == 'dual':
        actor = OmahaActor(seed,nS,nA,nB,network_params).to(device)
        critic = OmahaObsQCritic(seed,nS,nA,nB,network_params).to(device)
        if args.frozen:
            # Load pretrained hand recognizer
            copy_weights(actor,network_params['actor_hand_recognizer_path'])
            copy_weights(critic,network_params['critic_hand_recognizer_path'])
            # Expand conv1d over conv2d
            # expand_conv2d(actor,network_params['actor_hand_recognizer_path'])
            # expand_conv2d(critic,network_params['critic_hand_recognizer_path'])
        actor.summary
        critic.summary
        target_actor = OmahaActor(seed,nS,nA,nB,network_params).to(device)
        target_critic = OmahaObsQCritic(seed,nS,nA,nB,network_params).to(device)
        hard_update(actor,target_actor)
        hard_update(critic,target_critic)
        actor_optimizer = optim.Adam(actor.parameters(), lr=config.agent_params['actor_lr'],weight_decay=config.agent_params['L2'])
        critic_optimizer = optim.Adam(critic.parameters(), lr=config.agent_params['critic_lr'])
        actor_lrscheduler = StepLR(actor_optimizer, step_size=1, gamma=0.1)
        critic_lrscheduler = StepLR(critic_optimizer, step_size=1, gamma=0.1)
        learning_params['actor_optimizer'] = actor_optimizer
        learning_params['critic_optimizer'] = critic_optimizer
        learning_params['actor_lrscheduler'] = actor_lrscheduler
        learning_params['critic_lrscheduler'] = critic_lrscheduler
        # training loop
        # actor.share_memory()
        # critic.share_memory()
        processes = []
        # for debugging
        state,obs,done,action_mask,betsize_mask = env.reset()
        # check handboard
        print(state.shape)
        # net = ProcessHandBoard(network_params,hand_length=4)
        # handboard_input = torch.tensor(state[:,:,network_params['state_mapping']['hand_board']])
        # print('handboard_input',handboard_input)
        # net_out = net(handboard_input)
        # check preprocess
        # check actor
        print('actor')
        # actor(state,action_mask,betsize_mask)
        generate_trajectories(env,actor,critic,training_params,id=0)
        # actor,critic,learning_params = dual_learning_update(actor,critic,target_actor,target_critic,learning_params)
        # train_dual(env,actor,critic,target_actor,target_critic,training_params,learning_params,id=0)
        # for e in range(training_params['lr_steps']):
        #     for id in range(num_processes): # No. of processes
        #         p = mp.Process(target=train_dual, args=(env,actor,critic,target_actor,target_critic,training_params,learning_params,id))
        #         p.start()
        #         processes.append(p)
        #     for p in processes: 
        #         p.join()
        #     learning_params['actor_lrscheduler'].step()
        #     learning_params['critic_lrscheduler'].step()
        #     training_params['training_round'] = (e+1) * training_params['training_epochs']
        # save weights
        # torch.save(actor.state_dict(), os.path.join(config.agent_params['actor_path'],'OmahaActorFinal'))
        # torch.save(critic.state_dict(), os.path.join(config.agent_params['critic_path'],'OmahaCriticFinal'))
        print(f'Saved model weights to {os.path.join(path,"OmahaActorFinal")} and {os.path.join(path,"OmahaCriticFinal")}')
    else:
        raise ValueError(f'Network type {args.network_type} not supported')
    toc = time.time()
    print(f'Training completed in {(toc-tic)/60} minutes')