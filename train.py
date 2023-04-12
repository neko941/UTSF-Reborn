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
from models.LSTM import BiLSTM__Tensorflow
from utils.dataset import DatasetController
from utils.visualize import save_plot
from utils.general import increment_path

project = ROOT / 'runs'
name = 'exp'
overwrite = False
save_dir = str(increment_path(Path(project) / name, overwrite=overwrite, mkdir=True))

def main():
    path = r'.\configs\datasets\salinity-4_ids-split_column.yaml'
    granularity = 1440
    splitFeature = 'station'
    startTimeId = 0

    # path = r'.\configs\datasets\traffic-1_id-split_column.yaml'
    # splitFeature = 'current_geopath_id'  
    # granularity = 5 
    # startTimeId = 240

    # path = r'.\configs\datasets\weather_history-0_id-no_split_column.yaml'
    # splitFeature = None  
    # granularity = 60
    # startTimeId = 0

    lag = 1
    ahead = 1
    offset = 1
    workers = 8
    splitRatio = (0.7, 0.2, 0.1)
    seed = 941

    patience = 200
    optimizer = 'Adam'
    loss = 'MSE'
    epochs = 500
    learning_rate = 0.001
    batchsz = 64

    dataset = DatasetController(configsPath=path,
                                dirAsFeature=0,
                                granularity=granularity,
                                startTimeId=startTimeId,
                                splitFeature=splitFeature,
                                splitRatio=splitRatio,
                                workers=workers,
                                lag=lag, 
                                ahead=ahead, 
                                offset=offset,
                                savePath=save_dir).execute()
    X_train, y_train, X_val, y_val, X_test, y_test = dataset.GetData(shuffle=False)

    model_dict = [{ 
        'model' : BiLSTM__Tensorflow,
        'help' : '',
        'type' : 'Tensorflow',
        'config' : r'.\configs\models\DeepLearning\BiLSTM__Tensorflow.yaml'
    }]

    for item in model_dict:
        # if not vars(opt)[f'{item["model"].__name__}']: continue
        model = item['model'](input_shape=X_train.shape[-2:], 
                              output_shape=ahead, 
                              seed=seed,
                              normalize_layer=None,
                              modelConfigs=item.get('config'))
        model.build()
        model.fit(patience=patience, 
                  save_dir=save_dir, 
                  optimizer=optimizer, 
                  loss=loss, 
                  epochs=epochs, 
                  learning_rate=learning_rate, 
                  batchsz=batchsz,
                  X_train=X_train, y_train=y_train,
                  X_val=X_val, y_val=y_val)
        model.save(save_dir=save_dir, file_name=model.__class__.__name__)
        yhat = model.predict(X=X_test)
        scores = model.score(y=y_test, 
                             yhat=yhat, 
                             r=4,
                             path=save_dir)
        print(scores)
        if ahead == 1:
            visualize_path = os.path.join(save_dir, 'plots')
            os.makedirs(name=visualize_path, exist_ok=True)
            save_plot(filename=os.path.join(visualize_path, f'{model.__class__.__name__}.png'),
                        data=[{'data': [range(len(y_test)), y_test],
                                'color': 'green',
                                'label': 'y'},
                                {'data': [range(len(yhat)), yhat],
                                'color': 'red',
                                'label': 'yhat'}],
                        xlabel='Sample',
                        ylabel='Value')

if __name__ == '__main__':
    main()