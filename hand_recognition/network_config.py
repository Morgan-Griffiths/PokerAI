import datatypes as dt
from torch.nn import CrossEntropyLoss,BCELoss,SmoothL1Loss
from networks import *

class NetworkConfig(object):
    DataModels = {
            dt.DataTypes.FIVECARD : FiveCardClassification,
            dt.DataTypes.NINECARD : HandClassification,
            dt.DataTypes.TENCARD : TenCardClassificationV2,
            dt.DataTypes.THIRTEENCARD : ThirteenCardV2,
            dt.DataTypes.PARTIAL : PartialHandRegression,
            dt.DataTypes.BLOCKERS : BlockerClassification,
            dt.DataTypes.HANDRANKSFIVE : HandRankClassificationFC,
            dt.DataTypes.HANDRANKSNINE : HandRankClassificationNine,
            dt.DataTypes.SMALLDECK : SmalldeckClassification
        }
    LossFunctions = {
        dt.LearningCategories.MULTICLASS_CATEGORIZATION:CrossEntropyLoss,
        dt.LearningCategories.BINARY_CATEGORIZATION:BCELoss,
        dt.LearningCategories.REGRESSION:SmoothL1Loss,
    }