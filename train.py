from models.LSTM import BiLSTM__Tensorflow
from utils.dataset import DatasetController
import matplotlib.pyplot as plt 

def plot_difference(y, pred):
    plt.figure(figsize=(20, 6))
    times = range(len(y))
    y_to_plot = y.flatten()
    pred_to_plot = pred.flatten()

    plt.plot(times, y_to_plot, color='steelblue', marker='o', label='True value')
    plt.plot(times, pred_to_plot, color='orangered', marker='X', label='Prediction')

    plt.title('Adj Closing Price per day')
    plt.xlabel('Date')
    plt.ylabel('Adj Close Price')
    plt.legend()
    plt.show()

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

from utils.general import increment_path

project = ROOT / 'runs'
name = 'exp'
overwrite = False
save_dir = str(increment_path(Path(project) / name, overwrite=overwrite, mkdir=True))

path = r'.\configs\datasets\salinity-4_ids-split_column.yaml'
granularity = 1440
splitFeature = 'station'
startTimeId = 0

# path = r'.\configs\datasets\traffic-1_id-split_column.yaml'
# splitFeature = 'current_geopath_id'  
# granularity = 5 
# startTimeId=240

dataset = DatasetController(configsPath=path,
                            dirAsFeature=0,
                            granularity=granularity,
                            startTimeId=startTimeId,
                            splitFeature=splitFeature,
                            splitRatio=(0.7, 0.2, 0.1),
                            workers=8,
                            lag=1, 
                            ahead=1, 
                            offset=1,
                            savePath=save_dir).execute(shuffle=False)
X_train, y_train, X_val, y_val, X_test, y_test = dataset.X_train, dataset.y_train, dataset.X_val, dataset.y_val, dataset.X_test, dataset.y_test

model_dict = [{ 
    'model' : BiLSTM__Tensorflow,
    'help' : '',
    'type' : 'Tensorflow',
    'config' : '.\configs\models\DeepLearning\BiLSTM__Tensorflow.yaml'
}]
for item in model_dict:
    # if not vars(opt)[f'{item["model"].__name__}']: continue
    model = item['model'](input_shape=X_train.shape[-2:], 
                            output_shape=1, 
                            seed=941,
                            normalize_layer=None,
                            modelConfigs=item.get('config'))
    model.build()
    model.fit(patience=200, 
                save_dir=save_dir, 
                optimizer='Adam', 
                loss='MSE', 
                epochs=500, 
                learning_rate=0.001, 
                batchsz=64,
                X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val)
    model.save(save_dir=save_dir, file_name=model.__class__.__name__)
    yhat = model.predict(X=X_test)
    scores = model.score(y=y_test, 
                            yhat=yhat, 
                            r=4,
                            path=save_dir)
    print(scores)
    plot_difference(y_test[:300], yhat[:300])