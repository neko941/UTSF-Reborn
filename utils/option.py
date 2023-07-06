import argparse

""" Linear Machine Learning """
from models.MachineLearning import LinearRegressionWrapper
from models.MachineLearning import RidgeWrapper
from models.MachineLearning import RidgeCVWrapper
from models.MachineLearning import SGDRegressorWrapper
from models.MachineLearning import ElasticNetWrapper
from models.MachineLearning import ElasticNetCVWrapper
from models.MachineLearning import LarsWrapper
from models.MachineLearning import LarsCVWrapper
from models.MachineLearning import LassoWrapper
from models.MachineLearning import LassoCVWrapper
from models.MachineLearning import LassoLarsWrapper
from models.MachineLearning import LassoLarsCVWrapper
from models.MachineLearning import LassoLarsICWrapper
from models.MachineLearning import OrthogonalMatchingPursuitWrapper
from models.MachineLearning import OrthogonalMatchingPursuitCVWrapper
from models.MachineLearning import ARDRegressionWrapper
from models.MachineLearning import BayesianRidgeWrapper
from models.MachineLearning import HuberRegressorWrapper
from models.MachineLearning import QuantileRegressorWrapper
from models.MachineLearning import RANSACRegressorWrapper
from models.MachineLearning import TheilSenRegressorWrapper
from models.MachineLearning import PoissonRegressorWrapper
from models.MachineLearning import TweedieRegressorWrapper
from models.MachineLearning import GammaRegressorWrapper
from models.MachineLearning import DecisionTreeRegressorWrapper
from models.MachineLearning import ExtraTreeRegressorWrapper
from models.MachineLearning import LinearSVRWrapper
from models.MachineLearning import NuSVRWrapper
from models.MachineLearning import SVRWrapper

""" Ensemble Machine Learning """
from models.MachineLearning import ExtremeGradientBoostingRegression
from models.MachineLearning import AdaBoostRegressorWrapper
from models.MachineLearning import BaggingRegressorWrapper
from models.MachineLearning import ExtraTreesRegressorWrapper
from models.MachineLearning import GradientBoostingRegressorWrapper
from models.MachineLearning import IsolationForestWrapper
from models.MachineLearning import RandomForestRegressorWrapper
from models.MachineLearning import RandomTreesEmbeddingWrapper
from models.MachineLearning import StackingRegressorWrapper
from models.MachineLearning import VotingRegressorWrapper
from models.MachineLearning import HistGradientBoostingRegressorWrapper

""" Deep Learning """
from models.RNN import VanillaRNN__Tensorflow
from models.RNN import BiRNN__Tensorflow
from models.LSTM import VanillaLSTM__Tensorflow
from models.LSTM import BiLSTM__Tensorflow
from models.LSTM import Time2Vec_BiLSTM__Tensorflow
from models.LSTM import SelfAttention_BiLSTSM__Tensorflow
from models.LSTM import Multihead_BiLSTSM__Tensorflow
from models.GRU import VanillaGRU__Tensorflow
from models.GRU import BiGRU__Tensorflow
from models.LTSF_Linear import LTSF_Linear__Tensorflow
from models.LTSF_Linear import LTSF_NLinear__Tensorflow
from models.LTSF_Linear import LTSF_DLinear__Tensorflow
from models.Transformer import VanillaTransformer__Tensorflow
from models.RNN import EmbeddedRNN__Tensorflow
from models.LSTM import EmbeddedLSTM__Tensorflow
from models.LSTM import EmbeddedBiLSTM__Tensorflow
from models.LSTM import EmbeddedTime2Vec_BiLSTM__Tensorflow
from models.LSTM import EmbeddedMultihead_BiLSTSM__Tensorflow
from models.LSTM import EmbeddedSelfAttention_BiLSTSM__Tensorflow
from models.LTSF_Linear import LTSF_Embedded_Linear__Tensorflow
from models.LTSF_Linear import LTSF_Embedded_NLinear__Tensorflow
from models.LTSF_Linear import LTSF_Embedded_DLinear__Tensorflow

model_dict = [
    # Machine Learning Models
    {  
        'model'  : ExtremeGradientBoostingRegression,
        'help'   : '',
        'type'   : 'MachineLearning',
        'config' : r'.\configs\models\MachineLearning\ExtremeGradientBoostingRegression.yaml',
        'alias'  : ['XGBoost']
    },{
        'model'  : LinearRegressionWrapper,
        'help'   : '',
        'type'   : 'MachineLearning',
        'config' : r'.\configs\models\MachineLearning\LinearRegression.yaml',
        'alias'  : ['Linear']
    },{
        'model'  : RidgeWrapper,
        'help'   : '',
        'type'   : 'MachineLearning',
        'config' : r'.\configs\models\MachineLearning\Ridge.yaml',
        'alias'  : []
    },{
        'model'  : RidgeCVWrapper,
        'help'   : '',
        'type'   : 'MachineLearning',
        'config' : r'.\configs\models\MachineLearning\RidgeCV.yaml',
        'alias'  : []
    },{
        'model'  : SGDRegressorWrapper,
        'help'   : '',
        'type'   : 'MachineLearning',
        'config' : r'.\configs\models\MachineLearning\SGDRegressor.yaml',
        'alias'  : []
    },{
        'model'  : ElasticNetWrapper,
        'help'   : '',
        'type'   : 'MachineLearning',
        'config' : r'.\configs\models\MachineLearning\ElasticNet.yaml',
        'alias'  : []
    },{
        'model'  : ElasticNetCVWrapper,
        'help'   : '',
        'type'   : 'MachineLearning',
        'config' : r'.\configs\models\MachineLearning\ElasticNetCV.yaml',
        'alias'  : []
    },{
        'model'  : LarsWrapper,
        'help'   : '',
        'type'   : 'MachineLearning',
        'config' : r'.\configs\models\MachineLearning\Lars.yaml',
        'alias'  : []
    },{
        'model'  : LarsCVWrapper,
        'help'   : '',
        'type'   : 'MachineLearning',
        'config' : r'.\configs\models\MachineLearning\LarsCV.yaml',
        'alias'  : []
    },{
        'model'  : LassoWrapper,
        'help'   : '',
        'type'   : 'MachineLearning',
        'config' : r'.\configs\models\MachineLearning\Lasso.yaml',
        'alias'  : []
    },{
        'model'  : LassoCVWrapper,
        'help'   : '',
        'type'   : 'MachineLearning',
        'config' : r'.\configs\models\MachineLearning\LassoCV.yaml',
        'alias'  : []
    },{
        'model'  : LassoLarsWrapper,
        'help'   : '',
        'type'   : 'MachineLearning',
        'config' : r'.\configs\models\MachineLearning\LassoLars.yaml',
        'alias'  : []
    },{
        'model'  : LassoLarsCVWrapper,
        'help'   : '',
        'type'   : 'MachineLearning',
        'config' : r'.\configs\models\MachineLearning\LassoLarsCV.yaml',
        'alias'  : []
    },{
        'model'  : LassoLarsICWrapper,
        'help'   : '',
        'type'   : 'MachineLearning',
        'config' : r'.\configs\models\MachineLearning\LassoLarsIC.yaml',
        'alias'  : []
    },{
        'model'  : OrthogonalMatchingPursuitWrapper,
        'help'   : '',
        'type'   : 'MachineLearning',
        'config' : r'.\configs\models\MachineLearning\OrthogonalMatchingPursuit.yaml',
        'alias'  : []
    },{
        'model'  : OrthogonalMatchingPursuitCVWrapper,
        'help'   : '',
        'type'   : 'MachineLearning',
        'config' : r'.\configs\models\MachineLearning\OrthogonalMatchingPursuitCV.yaml',
        'alias'  : []
    },{
        'model'  : ARDRegressionWrapper,
        'help'   : '',
        'type'   : 'MachineLearning',
        'config' : r'.\configs\models\MachineLearning\ARDRegression.yaml',
        'alias'  : []
    },{
        'model'  : BayesianRidgeWrapper,
        'help'   : '',
        'type'   : 'MachineLearning',
        'config' : r'.\configs\models\MachineLearning\BayesianRidge.yaml',
        'alias'  : []
    },{
        'model'  : HuberRegressorWrapper,
        'help'   : '',
        'type'   : 'MachineLearning',
        'config' : r'.\configs\models\MachineLearning\HuberRegressor.yaml',
        'alias'  : []
    },{
        # 'model'  : QuantileRegressorWrapper,
        # 'help'   : '',
        # 'type'   : 'MachineLearning',
        # 'config' : r'.\configs\models\MachineLearning\QuantileRegressor.yaml',
        # 'alias'  : []
    # },{
        'model'  : RANSACRegressorWrapper,
        'help'   : '',
        'type'   : 'MachineLearning',
        'config' : r'.\configs\models\MachineLearning\RANSACRegressor.yaml',
        'alias'  : []
    },{
        'model'  : TheilSenRegressorWrapper,
        'help'   : '',
        'type'   : 'MachineLearning',
        'config' : r'.\configs\models\MachineLearning\TheilSenRegressor.yaml',
        'alias'  : []
    },{
        'model'  : PoissonRegressorWrapper,
        'help'   : '',
        'type'   : 'MachineLearning',
        'config' : r'.\configs\models\MachineLearning\PoissonRegressor.yaml',
        'alias'  : []
    },{
        'model'  : TweedieRegressorWrapper,
        'help'   : '',
        'type'   : 'MachineLearning',
        'config' : r'.\configs\models\MachineLearning\TweedieRegressor.yaml',
        'alias'  : []
    },{
        # 'model'  : GammaRegressorWrapper,
        # 'help'   : '',
        # 'type'   : 'MachineLearning',
        # 'config' : r'.\configs\models\MachineLearning\GammaRegressor.yaml',
        # 'alias'  : []
    # },{
        'model'  : DecisionTreeRegressorWrapper,
        'help'   : '',
        'type'   : 'MachineLearning',
        'config' : r'.\configs\models\MachineLearning\DecisionTreeRegressor.yaml',
        'alias'  : []
    },{
        'model'  : ExtraTreeRegressorWrapper,
        'help'   : '',
        'type'   : 'MachineLearning',
        'config' : r'.\configs\models\MachineLearning\ExtraTreeRegressor.yaml',
        'alias'  : []
    },{
        'model'  : LinearSVRWrapper,
        'help'   : '',
        'type'   : 'MachineLearning',
        'config' : r'.\configs\models\MachineLearning\LinearSVR.yaml',
        'alias'  : []
    },{
        'model'  : NuSVRWrapper,
        'help'   : '',
        'type'   : 'MachineLearning',
        'config' : r'.\configs\models\MachineLearning\NuSVR.yaml',
        'alias'  : []
    },{
        'model'  : SVRWrapper,
        'help'   : '',
        'type'   : 'MachineLearning',
        'config' : r'.\configs\models\MachineLearning\SVR.yaml',
        'alias'  : []
    },{
        'model'  : AdaBoostRegressorWrapper,
        'help'   : '',
        'type'   : 'MachineLearning',
        'config' : r'.\configs\models\MachineLearning\AdaBoostRegressor.yaml',
        'alias'  : []
    },{
        'model'  : BaggingRegressorWrapper,
        'help'   : '',
        'type'   : 'MachineLearning',
        'config' : r'.\configs\models\MachineLearning\BaggingRegressor.yaml',
        'alias'  : []
    },{
        'model'  : ExtraTreesRegressorWrapper,
        'help'   : '',
        'type'   : 'MachineLearning',
        'config' : r'.\configs\models\MachineLearning\ExtraTreesRegressor.yaml',
        'alias'  : []
    },{
        'model'  : GradientBoostingRegressorWrapper,
        'help'   : '',
        'type'   : 'MachineLearning',
        'config' : r'.\configs\models\MachineLearning\GradientBoostingRegressor.yaml',
        'alias'  : []
    },{
        'model'  : IsolationForestWrapper,
        'help'   : '',
        'type'   : 'MachineLearning',
        'config' : r'.\configs\models\MachineLearning\IsolationForest.yaml',
        'alias'  : []
    },{
        'model'  : RandomForestRegressorWrapper,
        'help'   : '',
        'type'   : 'MachineLearning',
        'config' : r'.\configs\models\MachineLearning\RandomForestRegressor.yaml',
        'alias'  : []
    },{
    #     'model'  : RandomTreesEmbeddingWrapper,
    #     'help'   : '',
    #     'type'   : 'MachineLearning',
    #     'config' : r'.\configs\models\MachineLearning\RandomTreesEmbedding.yaml',
    #     'alias'  : []
    # },{
        'model'  : StackingRegressorWrapper,
        'help'   : '',
        'type'   : 'MachineLearning',
        'config' : r'.\configs\models\MachineLearning\StackingRegressor.yaml',
        'alias'  : []
    },{
        'model'  : VotingRegressorWrapper,
        'help'   : '',
        'type'   : 'MachineLearning',
        'config' : r'.\configs\models\MachineLearning\VotingRegressor.yaml',
        'alias'  : []
    },{
        'model'  : HistGradientBoostingRegressorWrapper,
        'help'   : '',
        'type'   : 'MachineLearning',
        'config' : r'.\configs\models\MachineLearning\HistGradientBoostingRegressor.yaml',
        'alias'  : []
    },
    # Deep Learning Models
    {
        'model'  : VanillaRNN__Tensorflow,
        'help'   : '',
        'type'   : 'Tensorflow',
        'config' : r'.\configs\models\DeepLearning\VanillaRNN__Tensorflow.yaml',
        'alias'  : []
    },{ 
        'model'  : BiRNN__Tensorflow,
        'help'   : '',
        'type'   : 'Tensorflow',
        'config' : r'.\configs\models\DeepLearning\BiRNN__Tensorflow.yaml',
        'alias'  : []
    },{
        'model'  : EmbeddedRNN__Tensorflow,
        'help'   : '',
        'type'   : 'Tensorflow',
        'config' : r'.\configs\models\DeepLearning\VanillaRNN__Tensorflow.yaml',
        'alias'  : []
    },{ 
        'model'  : VanillaLSTM__Tensorflow,
        'help'   : '',
        'type'   : 'Tensorflow',
        'config' : r'.\configs\models\DeepLearning\VanillaLSTM__Tensorflow.yaml',
        'alias'  : []
    },{ 
        'model'  : EmbeddedLSTM__Tensorflow,
        'help'   : '',
        'type'   : 'Tensorflow',
        'config' : r'.\configs\models\DeepLearning\VanillaLSTM__Tensorflow.yaml',
        'alias'  : []
    },{ 
        'model'  : BiLSTM__Tensorflow,
        'help'   : '',
        'type'   : 'Tensorflow',
        'config' : r'.\configs\models\DeepLearning\BiLSTM__Tensorflow.yaml',
        'alias'  : []
    },{ 
        'model'  : EmbeddedBiLSTM__Tensorflow,
        'help'   : '',
        'type'   : 'Tensorflow',
        'config' : r'.\configs\models\DeepLearning\BiLSTM__Tensorflow.yaml',
        'alias'  : []
    },{ 
        'model'  : Time2Vec_BiLSTM__Tensorflow,
        'help'   : '',
        'type'   : 'Tensorflow',
        'config' : r'.\configs\models\DeepLearning\Time2Vec_BiLSTM__Tensorflow.yaml',
        'alias'  : []
    },{ 
        'model'  : EmbeddedSelfAttention_BiLSTSM__Tensorflow,
        'help'   : '',
        'type'   : 'Tensorflow',
        'config' : r'.\configs\models\DeepLearning\BiLSTM__Tensorflow.yaml',
        'alias'  : []
    },{ 
        'model'  : EmbeddedMultihead_BiLSTSM__Tensorflow,
        'help'   : '',
        'type'   : 'Tensorflow',
        'config' : r'.\configs\models\DeepLearning\BiLSTM__Tensorflow.yaml',
        'alias'  : []
    },{ 
        'model'  : EmbeddedTime2Vec_BiLSTM__Tensorflow,
        'help'   : '',
        'type'   : 'Tensorflow',
        'config' : r'.\configs\models\DeepLearning\Time2Vec_BiLSTM__Tensorflow.yaml',
        'alias'  : []
    },{ 
        'model'  : SelfAttention_BiLSTSM__Tensorflow,
        'help'   : '',
        'type'   : 'Tensorflow',
        'config' : r'.\configs\models\DeepLearning\SelfAttention_BiLSTSM__Tensorflow.yaml',
        'alias'  : []
    },{
        'model'  : Multihead_BiLSTSM__Tensorflow,
        'help'   : '',
        'type'   : 'Tensorflow',
        'config' : r'.\configs\models\DeepLearning\Multihead_BiLSTSM__Tensorflow.yaml',
        'alias'  : []
    },{
        'model'  : VanillaGRU__Tensorflow,
        'help'   : '',
        'type'   : 'Tensorflow',
        'config' : r'.\configs\models\DeepLearning\VanillaGRU__Tensorflow.yaml',
        'alias'  : []
    },{
        'model'  : BiGRU__Tensorflow,
        'help'   : '',
        'type'   : 'Tensorflow',
        'config' : r'.\configs\models\DeepLearning\BiGRU__Tensorflow.yaml',
        'alias'  : []
    },{
        'model'  : LTSF_Linear__Tensorflow,
        'help'   : '',
        'type'   : 'Tensorflow',
        'config' : r'.\configs\models\DeepLearning\LTSF_Linear__Tensorflow.yaml',
        'alias'  : []
    },{
        'model'  : LTSF_Embedded_Linear__Tensorflow,
        'help'   : '',
        'type'   : 'Tensorflow',
        'config' : r'.\configs\models\DeepLearning\LTSF_Linear__Tensorflow.yaml',
        'alias'  : []
    },{
        'model'  : LTSF_NLinear__Tensorflow,
        'help'   : '',
        'type'   : 'Tensorflow',
        'config' : r'.\configs\models\DeepLearning\LTSF_NLinear__Tensorflow.yaml',
        'alias'  : []
    },{
        'model'  : LTSF_Embedded_NLinear__Tensorflow,
        'help'   : '',
        'type'   : 'Tensorflow',
        'config' : r'.\configs\models\DeepLearning\LTSF_NLinear__Tensorflow.yaml',
        'alias'  : []
    },{
        'model'  : LTSF_DLinear__Tensorflow,
        'help'   : '',
        'type'   : 'Tensorflow',
        'config' : r'.\configs\models\DeepLearning\LTSF_DLinear__Tensorflow.yaml',
        'alias'  : []
    },{
        'model'  : LTSF_Embedded_DLinear__Tensorflow,
        'help'   : '',
        'type'   : 'Tensorflow',
        'config' : r'.\configs\models\DeepLearning\LTSF_DLinear__Tensorflow.yaml',
        'alias'  : []
    },{
        'model'  : VanillaTransformer__Tensorflow,
        'help'   : '',
        'type'   : 'Tensorflow',
        'config' : r'.\configs\models\DeepLearning\VanillaTransformer__Tensorflow.yaml',
        'alias'  : []
    }
]

def parse_opt(ROOT, known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10_000_000, help='total training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='')
    parser.add_argument('--batchsz', type=int, default=64, help='total batch size for all GPUs')
    parser.add_argument('--lag', type=int, default=5, help='')
    parser.add_argument('--ahead', type=int, default=1, help='')
    parser.add_argument('--offset', type=int, default=1, help='')
    parser.add_argument('--trainsz', type=float, default=0.7, help='')
    parser.add_argument('--valsz', type=float, default=0.1, help='')
    parser.add_argument('--resample', type=int, default=5, help='')

    # parser.add_argument('--granularity', type=int, default=1, help='by minutes')
    # parser.add_argument('--startTimeId', type=int, default=0, help='by minutes')

    parser.add_argument('--dataConfigs', default='data.yaml', help='dataset')
    parser.add_argument('--crossValidation', action='store_true', help='')
    parser.add_argument('--patience', type=int, default=1000, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--project', default=ROOT / 'runs', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--overwrite', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--cyclicalPattern', action='store_true', help='')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW', 'Nadam', 'RMSprop', 'Adafactor', 'Adadelta', 'Adagrad', 'Adamax', 'Ftrl'], default='Adam', help='optimizer')
    parser.add_argument('--polarsFilling', type=str, choices=[None, 'bfill', 'ffill', 'min', 'max', 'mean', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'barycentric', 'polynomial'], default=None, help='')
    parser.add_argument('--machineFilling', type=str, choices=[None, 'XGBoost'], default=None, help='')
    parser.add_argument('--loss', type=str, choices=['MSE'], default='MSE', help='losses')
    parser.add_argument('--seed', type=int, default=941, help='Global training seed')
    parser.add_argument('--round', type=int, default=-1, help='Round decimals in results, -1 to disable')
    parser.add_argument('--individual', action='store_true', help='for LTSF Linear models')
    parser.add_argument('--debug', action='store_true', help='print debug information in table')
    parser.add_argument('--multimodels', action='store_true', help='split data of n segment ids for n models ')
    parser.add_argument('--workers', type=int, default=8, help='')
    parser.add_argument('--low_memory', action='store_true', help='Ultilize disk')
    parser.add_argument('--normalization', action='store_true', help='')

    parser.add_argument('--dirFeatureName', type=str, default='dir', help='')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    parser.add_argument('--all', action='store_true', help='Use all available models')
    parser.add_argument('--MachineLearning', action='store_true', help='')
    parser.add_argument('--DeepLearning', action='store_true', help='')
    parser.add_argument('--Tensorflow', action='store_true', help='')
    parser.add_argument('--Pytorch', action='store_true', help='')
    parser.add_argument('--LTSF', action='store_true', help='Using all LTSF Linear Models')

    for item in model_dict:
        parser.add_argument(f"--{item['model'].__name__}", action='store_true', help=f"{item['help']}")
        # for flag in [item['model'].__name__, *item['alias'].__name__]:
        #     parser.add_argument(f"--{flag}", action='store_true', help=f"{item['help']}")

    return parser.parse_known_args()[0] if known else parser.parse_args()

def update_opt(opt):
    if opt.all:
        opt.MachineLearning = True
        opt.DeepLearning = True
    if opt.DeepLearning:
        opt.Tensorflow = True
        opt.Pytorch = True
    if opt.LTSF:
        opt.LTSF_Linear__Tensorflow = True
        opt.LTSF_NLinear__Tensorflow = True
        opt.LTSF_DLinear__Tensorflow = True
    for item in model_dict:
        if any([opt.Tensorflow and item['type']=='Tensorflow',
                opt.Pytorch and item['type']=='Pytorch',
                opt.MachineLearning and item['type']=='MachineLearning']): 
            vars(opt)[f'{item["model"].__name__}'] = True

    if opt.machineFilling: 
        opt.cyclicalPattern = False
        opt.polarsFilling = None
    return opt