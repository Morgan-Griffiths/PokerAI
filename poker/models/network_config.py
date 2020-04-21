import hand_recognition.datatypes as dt
from models.networks import FiveCardClassification,ThirteenCard,ThirteenCardV2,TenCardClassificationV2,HandClassification
from torch.nn import CrossEntropyLoss,BCELoss,SmoothL1Loss

class NetworkConfig(object):
    DataModels = {
            dt.DataTypes.FIVECARD : FiveCardClassification,
            dt.DataTypes.NINECARD : HandClassification,
            dt.DataTypes.TENCARD : TenCardClassificationV2,
            dt.DataTypes.THIRTEENCARD : ThirteenCard,
            dt.DataTypes.PARTIAL : 'Not implemented',
            dt.DataTypes.BLOCKERS : 'Not implemented'
        }
    LossFunctions = {
        dt.LearningCategories.MULTICLASS_CATEGORIZATION:CrossEntropyLoss,
        dt.LearningCategories.BINARY_CATEGORIZATION:BCELoss,
        dt.LearningCategories.REGRESSION:SmoothL1Loss,
    }