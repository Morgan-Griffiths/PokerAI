import poker_env.datatypes as dt
from models.networks import *

class CriticType:
    Q='q'
    REG='reg'

class NetworkConfig(object):
    EnvModels = {
            dt.GameTypes.HOLDEM : {
                'actor':HoldemBaseline,
                'critic':{
                    CriticType.Q : HoldemQCritic,
                    CriticType.REG : HoldemBaselineCritic
                }
            },
            dt.GameTypes.OMAHAHI : {
                'actor':HoldemBaseline,
                'critic':{
                    CriticType.Q : HoldemQCritic,
                    CriticType.REG : HoldemBaselineCritic
                }
            }
        }