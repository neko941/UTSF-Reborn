import os
import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import matplotlib
matplotlib.use('Agg') # Tcl_AsyncDelete: async handler deleted by the wrong thread

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR) # disable absl INFO and WARNING log messages

import gc
import shutil
import numpy as np
import matplotlib.pyplot as plt 
from utils.dataset import DatasetController
from utils.visualize import save_plot

from utils.general import set_seed
from utils.general import yaml_save
from utils.general import increment_path

from utils.option import parse_opt
from utils.option import update_opt
from utils.option import model_dict

from utils.metrics import metric_dict
from utils.rich2polars import table_to_df
from utils.activations import get_custom_activations

from rich import box as rbox
from rich.table import Table
from rich.console import Console
from rich.terminal_theme import MONOKAI

from utils.general import convert_seconds
import csv

from rich.progress import Progress
from rich.progress import BarColumn 
from rich.progress import TextColumn
from rich.progress import TimeElapsedColumn
from rich.progress import MofNCompleteColumn
from rich.progress import TimeRemainingColumn

def main(opt):
    """ Get the save directory for this run """
    save_dir = str(increment_path(Path(opt.project) / opt.name, overwrite=opt.overwrite, mkdir=True))
    # save_dir = str(Path(opt.project) / 'crossvalidation-data_avg_min_max_std_rain-all_ids-use_avg_min_max_std_rain-fill_ffill' / f'r{opt.resample}l{opt.lag}')

    """ Path to save configs """
    path_configs = Path(save_dir, 'configs')
    path_configs.mkdir(parents=True, exist_ok=True)

    """ Set seed """
    opt.seed = set_seed(opt.seed)

    """ Add custom function """
    get_custom_activations()

    """ Save init options """
    yaml_save(path_configs / 'opt.yaml', vars(opt))

    """ Update options """
    opt = update_opt(opt)
    shuffle = False
    opt.NuSVRWrapper = False
    opt.SVRWrapper = False
    opt.HistGradientBoostingRegressorWrapper = False
    
    opt.EmbeddedRNN__Tensorflow = False
    opt.EmbeddedLSTM__Tensorflow = False
    opt.EmbeddedBiLSTM__Tensorflow = False
    opt.EmbeddedTime2Vec_BiLSTM__Tensorflow = False
    opt.EmbeddedMultihead_BiLSTSM__Tensorflow = False
    opt.EmbeddedSelfAttention_BiLSTSM__Tensorflow = False
    opt.LTSF_Embedded_Linear__Tensorflow = False
    opt.LTSF_Embedded_NLinear__Tensorflow = False
    opt.LTSF_Embedded_DLinear__Tensorflow = False
    # CrossValidation = True
    CrossValidation = False

    """ Save updated options """
    yaml_save(path_configs / 'updated_opt.yaml', vars(opt))
    shutil.copyfile(opt.dataConfigs, path_configs/os.path.basename(opt.dataConfigs))

    """ Preprocessing dataset """
    dataset = DatasetController(configsPath=opt.dataConfigs,
                                resample=opt.resample,
                                # startTimeId=opt.startTimeId,
                                splitRatio=(opt.trainsz, opt.valsz, 1-opt.trainsz-opt.valsz),
                                workers=opt.workers,
                                lag=opt.lag, 
                                ahead=opt.ahead, 
                                offset=opt.offset,
                                savePath=save_dir,
                                polarsFilling=opt.polarsFilling,
                                machineFilling=opt.machineFilling,
                                low_memory=opt.low_memory,
                                normalization=opt.normalization,
                                cyclicalPattern=opt.cyclicalPattern).execute()
    X_train, y_train, X_val, y_val, X_test, y_test = dataset.GetData(shuffle=shuffle)
    scaler = dataset.scaler
    del dataset
    gc.collect()
    
    """ Create result table """
    console = Console(record=True)
    table = Table(title="[cyan]Results", show_header=True, header_style="bold magenta", box=rbox.ROUNDED, show_lines=True)
    [table.add_column(f'[green]{name}', justify='center') for name in ['Name', 'Time', *list(metric_dict.keys())]]

    

    if not CrossValidation:
        for item in model_dict:
            if not vars(opt)[f'{item["model"].__name__}']: continue
            shutil.copyfile(item['config'], path_configs/os.path.basename(item['config']))
            datum = train(model=item['model'], 
                        modelConfigs=item['config'], 
                        data=[[X_train, y_train], [X_val, y_val], [X_test, y_test]], 
                        save_dir=save_dir,
                        ahead=opt.ahead, 
                        seed=opt.seed, 
                        normalize_layer=None,
                        learning_rate=opt.lr,
                        epochs=opt.epochs, 
                        patience=opt.patience,
                        optimizer=opt.optimizer, 
                        loss=opt.loss,
                        batchsz=opt.batchsz,
                        r=opt.round,
                        enc_in=1,
                        scaler=scaler,
                        time_as_int=False)
            table.add_row(*datum)
            console.print(table)
            console.save_svg(os.path.join(save_dir, 'results.svg'), theme=MONOKAI)  
            table_to_df(table).write_csv(os.path.join(save_dir, 'results.csv'))
    else:
        x = np.concatenate([X_train, X_val, X_test], axis=0)
        x = np.array_split(x, 100, axis=0)
        y = np.concatenate([y_train, y_val, y_test], axis=0)
        y = np.array_split(y, 100, axis=0)

        
        f = open(os.path.join(save_dir, 'results.csv'), 'w', newline='', encoding='utf-8')
        writer = csv.writer(f)
        
        values_dir = Path(save_dir) / 'values'
        values_dir.mkdir(parents=True, exist_ok=True)
        
        for item in model_dict:
            if not vars(opt)[f'{item["model"].__name__}']: continue
            shutil.copyfile(item['config'], path_configs/os.path.basename(item['config']))
            data = []
            name = ''
            with Progress("[bright_cyan][progress.description]{task.description}",
                              BarColumn(),
                              TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                              TextColumn("•Items"),
                              MofNCompleteColumn(), # "{task.completed}/{task.total}",
                              TextColumn("•Remaining"),
                              TimeRemainingColumn(),
                              TextColumn("•Total"),
                              TimeElapsedColumn()) as progress:
                # start = int((1- opt.trainsz - opt.valsz)*100)
                start = 30
                for i in progress.track(range(start, 
                                              int(opt.trainsz*100)+1,
                                              10), description='Cross Validate'):
                    X_train = np.concatenate(x[:i])
                    y_train = np.concatenate(y[:i])
                    # X_val = np.concatenate(x[i:int(i + opt.valsz*100)]) 
                    # y_val = np.concatenate(y[i:int(i + opt.valsz*100)]) 
                    # X_test = np.concatenate(x[int(i + opt.valsz*100) : int(int(i + opt.valsz*100) + (1- opt.trainsz - opt.valsz)*100)])
                    # y_test = np.concatenate(y[int(i + opt.valsz*100) : int(int(i + opt.valsz*100) + (1- opt.trainsz - opt.valsz)*100)])
                    
                    X_val = X_test = np.concatenate(x[int(i + opt.valsz*100):int(int(i + opt.valsz*100) + (1- opt.trainsz - opt.valsz)*100)]) 
                    y_val = y_test = np.concatenate(y[int(i + opt.valsz*100):int(int(i + opt.valsz*100) + (1- opt.trainsz - opt.valsz)*100)])
                   
                    datum = train(model=item['model'], 
                                  modelConfigs=item['config'], 
                                  data=[[X_train, y_train], [X_val, y_val], [X_test, y_test]], 
                                  save_dir=save_dir,
                                  ahead=opt.ahead, 
                                  seed=opt.seed, 
                                  normalize_layer=None,
                                  learning_rate=opt.lr,
                                  epochs=opt.epochs, 
                                  patience=opt.patience,
                                  optimizer=opt.optimizer, 
                                  loss=opt.loss,
                                  batchsz=opt.batchsz,
                                  r=opt.round,
                                  enc_in=1,
                                  scaler=scaler,
                                  time_as_int=True)
                    name = datum[0]
                    data.append([float(ele) for ele in datum[1:]])
                    datum[0] = datum[0] + f'_{i}'
                    writer.writerow(datum)
                    np.save(values_dir / f'xtrain_{name}_{i}.npy', X_train)
                    np.save(values_dir / f'ytrain_{name}_{i}.npy', y_train)
                    np.save(values_dir / f'xval_{name}_{i}.npy', X_val)
                    np.save(values_dir / f'yval_{name}_{i}.npy', y_val)
                    np.save(values_dir / f'xtest_{name}_{i}.npy', X_test)
                    np.save(values_dir / f'ytest_{name}_{i}.npy', y_test)
            
            
            data = np.array(data)
            data = [name, convert_seconds(np.sum(data[:, 0])), *np.mean(data[:, 1:], axis=0)]
            data = [str(ele) for ele in data]
            table.add_row(*data)
            console.print(table)
            console.save_svg(os.path.join(save_dir, 'results.svg'), theme=MONOKAI)  
            # table_to_df(table).write_csv()
        f.close()

def train(model, modelConfigs, data, save_dir, ahead,
          seed: int = 941, 
          normalize_layer=None,
          learning_rate: float = 1e-3,
          epochs: int = 10_000_000, 
          patience: int = 1_000,
          optimizer:str = 'Adam', 
          loss:str = 'MSE',
          batchsz:int = 64,
          r: int = 4,
          enc_in: int = 1,
          scaler = None,
          time_as_int:bool = False) -> list:
    # import tensorflow as tf
    # model = tf.keras.models.load_model('VanillaLSTM__Tensorflow')
    # model.summary()

    model = model(input_shape=data[0][0].shape[-2:],
                  modelConfigs=modelConfigs, 
                  output_shape=ahead, 
                  seed=seed,
                  normalize_layer=None,
                  save_dir=save_dir,
                  enc_in=enc_in)
    model.build()
    # model.model.built = True
    # model.load('LTSF_Linear__Tensorflow_bestckpt.index')
    model.fit(patience=patience, 
              optimizer=optimizer, 
              loss=loss, 
              epochs=epochs, 
              learning_rate=learning_rate, 
              batchsz=batchsz,
              X_train=data[0][0], y_train=data[0][1],
              X_val=data[1][0], y_val=data[1][1],
              time_as_int=time_as_int)
    model.save(file_name=f'{model.__class__.__name__}')
    

    weight=os.path.join(save_dir, 'weights', f"{model.__class__.__name__}_best.h5")
    if not os.path.exists(weight): weight = model.save(file_name=model.__class__.__name__)
    else: model.save(save_dir=save_dir, file_name=model.__class__.__name__)
    # weight = r'runs\exp809\weights\VanillaLSTM__Tensorflow_best.h5'
    if weight is not None: model.load(weight)

    # predict values
    # yhat = model.predict(X=data[2][0][:10_000], name='test')
    # ytrainhat = model.predict(X=data[0][0][:10_000], name='train')
    # yvalhat = model.predict(X=data[1][0][:10_000], name='val')
    yhat = model.predict(X=data[2][0], scaler=scaler)
    ytrainhat = model.predict(X=data[0][0], scaler=scaler)
    yvalhat = model.predict(X=data[1][0], scaler=scaler)

    # calculate scores
    # print(data[2][1], yhat)
    # print(yhat[:10])
    # print(data[2][1][:10])
    scores = model.score(y=data[2][1], 
                         yhat=yhat, 
                         # path=save_dir,
                         r=r)

    # plot values
    model.plot(save_dir=save_dir, y=data[0][1], yhat=ytrainhat, dataset='Train')
    model.plot(save_dir=save_dir, y=data[1][1], yhat=yvalhat, dataset='Val')
    model.plot(save_dir=save_dir, y=data[2][1], yhat=yhat, dataset='Test')
    
    model.plot(save_dir=save_dir, y=data[0][1][:100], yhat=ytrainhat[:100], dataset='Train100')
    model.plot(save_dir=save_dir, y=data[1][1][:100], yhat=yvalhat[:100], dataset='Val100')
    model.plot(save_dir=save_dir, y=data[2][1][:100], yhat=yhat[:100], dataset='Test100')

    return [model.__class__.__name__, model.time_used, *scores]

def run(**kwargs):
    """ 
    Usage (example)
        import train
        train.run(all=True, 
                  configsPath=data.yaml,
                  lag=5,
                  ahead=1,
                  offset=1)
    """
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt

if __name__ == "__main__":
    opt = parse_opt(ROOT=ROOT)
    main(opt)