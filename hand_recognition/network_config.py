import datatypes as dt
from torch.nn import CrossEntropyLoss,BCELoss,SmoothL1Loss
from networks import HandRankClassificationFC,HandRankClassificationNine,FiveCardClassification,HandClassification,TenCardClassificationV2,ThirteenCardV2,PartialHandRegression,BlockerClassification,HandRankClassificationFive,HandBoard

class NetworkConfig(object):
    DataModels = {
            dt.DataTypes.FIVECARD : FiveCardClassification,
            dt.DataTypes.NINECARD : HandClassification,
            dt.DataTypes.TENCARD : TenCardClassificationV2,
            dt.DataTypes.THIRTEENCARD : ThirteenCardV2,
            dt.DataTypes.PARTIAL : PartialHandRegression,
            dt.DataTypes.BLOCKERS : BlockerClassification,
            dt.DataTypes.HANDRANKSFIVE : HandRankClassificationFive,
            dt.DataTypes.HANDRANKSNINE : HandBoard
        }
    LossFunctions = {
        dt.LearningCategories.MULTICLASS_CATEGORIZATION:CrossEntropyLoss,
        dt.LearningCategories.BINARY_CATEGORIZATION:BCELoss,
        dt.LearningCategories.REGRESSION:SmoothL1Loss,
    }