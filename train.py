import os
import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR) # disable absl INFO and WARNING log messages

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

def main(opt):
    """ Get the save directory for this run """
    save_dir = str(increment_path(Path(opt.project) / opt.name, overwrite=opt.overwrite, mkdir=True))

    """ Set seed """
    opt.seed = set_seed(opt.seed)

    """ Add custom function """
    get_custom_activations()

    """ Save init options """
    yaml_save(os.path.join(save_dir, 'opt.yaml'), vars(opt))

    """ Update options """
    opt = update_opt(opt)

    """ Fixed config for testing """
    # opt.ExtremeGradientBoostingRegression = True
    # opt.LTSF_NLinear__Tensorflow = True
    # opt.BiLSTM__Tensorflow = True
    # opt.machineFilling = 'XGBoost'
    
    # opt.dataConfigs = r'.\configs\datasets\salinity-615_csv-lag5-ahead1-offset1.yaml'
    # opt.granularity = None
    # opt.startTimeId = None

    # opt.dataConfigs = r'.\configs\datasets\salinity-1_id-split_column.yaml'
    # opt.granularity = 1440
    # opt.startTimeId = 0

    # opt.dataConfigs = r'.\configs\datasets\salinity-4_ids-split_column.yaml'
    # opt.granularity = 1440
    # opt.startTimeId = 0

    # opt.dataConfigs= r'.\configs\datasets\traffic-1_id-split_column.yaml'
    # opt.granularity = 5 
    # opt.startTimeId = 240

    # path = r'.\configs\datasets\weather_history-0_id-no_split_column.yaml'
    # granularity = 60
    # startTimeId = 0

    shuffle = False

    """ Save updated options """
    yaml_save(os.path.join(save_dir, 'updated_opt.yaml'), vars(opt))

    """ Preprocessing dataset """
    dataset = DatasetController(configsPath=opt.dataConfigs,
                                granularity=opt.granularity,
                                startTimeId=opt.startTimeId,
                                splitRatio=(opt.trainsz, opt.valsz, 1-opt.trainsz-opt.valsz),
                                workers=opt.workers,
                                lag=opt.lag, 
                                ahead=opt.ahead, 
                                offset=opt.offset,
                                savePath=save_dir,
                                polarsFilling=opt.polarsFilling,
                                machineFilling=opt.machineFilling).execute(cyclicalPattern=opt.cyclicalPattern)
    X_train, y_train, X_val, y_val, X_test, y_test = dataset.GetData(shuffle=shuffle)

    """ Create result table """
    console = Console(record=True)
    table = Table(title="[cyan]Results", show_header=True, header_style="bold magenta", box=rbox.ROUNDED, show_lines=True)
    [table.add_column(f'[green]{name}', justify='center') for name in ['Name', 'Time', *list(metric_dict.keys())]]

    """ Train models """
    for item in model_dict:
        if not vars(opt)[f'{item["model"].__name__}']: continue
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
                      enc_in=1)
        table.add_row(*datum)
        console.print(table)
        console.save_svg(os.path.join(save_dir, 'results.svg'), theme=MONOKAI)  
    table_to_df(table).write_csv(os.path.join(save_dir, 'results.csv'))

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
          enc_in: int = 1) -> list:
    model = model(input_shape=data[0][0].shape[-2:],
                  modelConfigs=modelConfigs, 
                  output_shape=ahead, 
                  seed=seed,
                  normalize_layer=None,
                  save_dir=save_dir,
                  enc_in=enc_in)
    model.build()
    model.fit(patience=patience, 
              optimizer=optimizer, 
              loss=loss, 
              epochs=epochs, 
              learning_rate=learning_rate, 
              batchsz=batchsz,
              X_train=data[0][0], y_train=data[0][1],
              X_val=data[1][0], y_val=data[1][1])
    model.save(file_name=f'{model.__class__.__name__}')
    
    # predict values
    yhat = model.predict(X=data[2][0])
    ytrainhat = model.predict(X=data[0][0])
    yvalhat = model.predict(X=data[1][0])

    # calculate scores
    scores = model.score(y=data[2][1], yhat=yhat, r=r, path=save_dir)

    # plot values
    model.plot(save_dir=save_dir, y=data[0][1], yhat=ytrainhat, dataset='Train')
    model.plot(save_dir=save_dir, y=data[1][1], yhat=yvalhat, dataset='Val')
    model.plot(save_dir=save_dir, y=data[2][1], yhat=yhat, dataset='Test')

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