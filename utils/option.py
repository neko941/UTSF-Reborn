import argparse

from models.LSTM import BiLSTM__Tensorflow
from models.Transformer import VanillaTransformer__Tensorflow

model_dict = [
    { 
     'model' : BiLSTM__Tensorflow,
     'help' : '',
     'type' : 'Tensorflow',
     'config' : r'.\configs\models\DeepLearning\BiLSTM__Tensorflow.yaml'
    },{
     'model' : VanillaTransformer__Tensorflow,
     'help' : '',
     'type' : 'Tensorflow',
     'config' : r'.\configs\models\DeepLearning\VanillaTransformer__Tensorflow.yaml'
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
    parser.add_argument('--valsz', type=float, default=0.2, help='')

    parser.add_argument('--granularity', type=int, default=1, help='by minutes')
    parser.add_argument('--startTimeId', type=int, default=0, help='by minutes')

    parser.add_argument('--dataConfigs', default='data.yaml', help='dataset')
    parser.add_argument('--patience', type=int, default=1000, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--project', default=ROOT / 'runs', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--overwrite', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--cyclicalPattern', action='store_true', help='')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW', 'Nadam', 'RMSprop', 'Adafactor', 'Adadelta', 'Adagrad', 'Adamax', 'Ftrl'], default='Adam', help='optimizer')
    parser.add_argument('--filling', type=str, choices=[None, 'forward', 'backward', 'min', 'max', 'mean'], default=None, help='optimizer')
    parser.add_argument('--loss', type=str, choices=['MSE'], default='MSE', help='losses')
    parser.add_argument('--seed', type=int, default=941, help='Global training seed')
    parser.add_argument('--round', type=int, default=-1, help='Round decimals in results, -1 to disable')
    parser.add_argument('--individual', action='store_true', help='for LTSF Linear models')
    parser.add_argument('--debug', action='store_true', help='print debug information in table')
    parser.add_argument('--multimodels', action='store_true', help='split data of n segment ids for n models ')
    parser.add_argument('--workers', type=int, default=1, help='')

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
    return opt