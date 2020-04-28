import hand_recognition.datatypes as dt
import poker.datatypes as pdt
from models.networks import *
from torch.nn import CrossEntropyLoss,BCELoss,SmoothL1Loss

class CriticType:
    Q='q'
    REG='reg'

class NetworkConfig(object):
    DataModels = {
            dt.DataTypes.FIVECARD : FiveCardClassification,
            dt.DataTypes.NINECARD : HandClassification,
            dt.DataTypes.TENCARD : TenCardClassificationV2,
            dt.DataTypes.THIRTEENCARD : ThirteenCard,
            dt.DataTypes.PARTIAL : PartialHandRegression,
            dt.DataTypes.BLOCKERS : BlockerClassification,
            dt.DataTypes.HANDRANKS : HandRankClassification
        }
    LossFunctions = {
        dt.LearningCategories.MULTICLASS_CATEGORIZATION:CrossEntropyLoss,
        dt.LearningCategories.BINARY_CATEGORIZATION:BCELoss,
        dt.LearningCategories.REGRESSION:SmoothL1Loss,
    }
    EnvModels = {
            pdt.GameTypes.KUHN : Baseline,
            pdt.GameTypes.COMPLEXKUHN : Baseline,
            pdt.GameTypes.HOLDEM : HoldemBaseline,
            pdt.GameTypes.OMAHAHI : 'Not implemented',
        }
    CriticModels = {
        CriticType.Q : BaselineCritic,
        CriticType.REG : BaselineKuhnCritic
    }
    ActorModels = {
        CriticType.Q : Baseline,
        CriticType.REG : Baseline
    }