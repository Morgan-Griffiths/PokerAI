class Config(object):
    def __init__(self):
        self.act_dict = {'SB':0,'BB':1}
        self.agent = 'actor_critic'
        self.maxlen = 10
        self.training_params = {
                'epochs':2500,
                'training_round':0,
                'save_dir':'checkpoints/RL'
            }
        self.agent_params = {
            'BUFFER_SIZE':10000,
            'MIN_BUFFER_SIZE':200,
            'BATCH_SIZE':50,
            'ALPHA':0.6, # 0.7 or 0.6,
            'START_BETA':0.5, # from 0.5-1,
            'END_BETA':1,
            'LEARNING_RATE':0.00025,
            'EPSILON':1,
            'MIN_EPSILON':0.01,
            'GAMMA':0.99,
            'TAU':0.01,
            'UPDATE_EVERY':4,
            'CLIP_NORM':10,
            'L2': 0.01,
            'embedding':True,
            'network':None,
            'critic_network':None,
            'actor_network':None,
            'critic_type':'q',
            'embedding_size': 32,
            'actor_lr':1e-4,
            'critic_lr':4e-7
        }