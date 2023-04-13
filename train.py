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
from utils.option import model_dict
from utils.visualize import save_plot
from utils.general import increment_path

project = ROOT / 'runs'
name = 'exp'
overwrite = False
save_dir = str(increment_path(Path(project) / name, overwrite=overwrite, mkdir=True))

def train(model, modelConfigs, data, save_dir, ahead,
          seed=941, 
          normalize_layer=None,
          learning_rate=1e-3,
          epochs=10_000_000, 
          patience=1_000,
          optimizer='Adam', 
          loss='MSE',
          batchsz=64):
    model = model(input_shape=data[0][0].shape[-2:],
                  modelConfigs=modelConfigs, 
                  output_shape=ahead, 
                  seed=seed,
                  normalize_layer=None)
    model.build()
    model.fit(patience=patience, 
              save_dir=save_dir, 
              optimizer=optimizer, 
              loss=loss, 
              epochs=epochs, 
              learning_rate=learning_rate, 
              batchsz=batchsz,
              X_train=data[0][0], y_train=data[0][1],
              X_val=data[1][0], y_val=data[1][1])
    model.save(save_dir=save_dir, file_name=model.__class__.__name__)
    yhat = model.predict(X=data[2][0])
    ytrainhat = model.predict(X=data[0][0])
    yvalhat = model.predict(X=data[1][0])
    scores = model.score(y=data[2][1], 
                         yhat=yhat, 
                         r=4,
                         path=save_dir)
    print(scores)
    if ahead == 1:
        visualize_path = os.path.join(save_dir, 'plots')
        os.makedirs(name=visualize_path, exist_ok=True)

        save_plot(filename=os.path.join(visualize_path, f'{model.__class__.__name__}-Test.png'),
                  data=[{'data': [range(len(data[2][1])), data[2][1]],
                         'color': 'green',
                         'label': 'y'},
                        {'data': [range(len(yhat)), yhat],
                         'color': 'red',
                         'label': 'yhat'}],
                  xlabel='Sample',
                  ylabel='Value')

        save_plot(filename=os.path.join(visualize_path, f'{model.__class__.__name__}-Train.png'),
                  data=[{'data': [range(len(data[0][1])), data[0][1]],
                         'color': 'green',
                         'label': 'y'},
                        {'data': [range(len(ytrainhat)), ytrainhat],
                         'color': 'red',
                         'label': 'yhat'}],
                  xlabel='Sample',
                  ylabel='Value')

        save_plot(filename=os.path.join(visualize_path, f'{model.__class__.__name__}-Val.png'),
                  data=[{'data': [range(len(data[1][1])), data[1][1]],
                         'color': 'green',
                         'label': 'y'},
                        {'data': [range(len(yvalhat)), yvalhat],
                         'color': 'red',
                         'label': 'yhat'}],
                  xlabel='Sample',
                  ylabel='Value')

def main():
    path = r'.\configs\datasets\salinity-615_csv-lag5-ahead1-offset1.yaml'
    granularity = None
    startTimeId = None

    # path = r'.\configs\datasets\salinity-1_id-split_column.yaml'
    # granularity = 1440
    # startTimeId = 0

    # path = r'.\configs\datasets\traffic-1_id-split_column.yaml'
    # granularity = 5 
    # startTimeId = 240

    # path = r'.\configs\datasets\weather_history-0_id-no_split_column.yaml'
    # granularity = 60
    # startTimeId = 0

    lag = 5
    ahead = 1
    offset = 1
    workers = 8
    splitRatio = (0.7, 0.2, 0.1)
    seed = 941
    cyclicalPattern = False
    patience = 1_000
    optimizer = 'Adam'
    loss = 'MSE'
    epochs = 10_000_000
    learning_rate = 0.001
    batchsz = 64

    dataset = DatasetController(configsPath=path,
                                granularity=granularity,
                                startTimeId=startTimeId,
                                splitRatio=splitRatio,
                                workers=workers,
                                lag=lag, 
                                ahead=ahead, 
                                offset=offset,
                                savePath=save_dir).execute(cyclicalPattern=cyclicalPattern)
    X_train, y_train, X_val, y_val, X_test, y_test = dataset.GetData(shuffle=False)

    for item in model_dict:
        # if not vars(opt)[f'{item["model"].__name__}']: continue
        train(model=item['model'], 
              modelConfigs=item['config'], 
              data=[[X_train, y_train], [X_val, y_val], [X_test, y_test]], 
              save_dir=save_dir,
              ahead=ahead, 
              seed=seed, 
              normalize_layer=None,
              learning_rate=learning_rate,
              epochs=epochs, 
              patience=patience,
              optimizer=optimizer, 
              loss=loss,
              batchsz=batchsz)

def run(**kwargs):
    """ 
    Usage (example)
        import main
        main.run(all=True, 
                 source=data.yaml,
                 Normalization=True)
    """
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt

if __name__ == '__main__':
    main()