import os
from abc import abstractmethod
import json
import time

import tensorflow as tf

from keras.optimizers import SGD
from keras.optimizers import Ftrl
from keras.optimizers import Adam
from keras.optimizers import Nadam
from keras.optimizers import RMSprop
from keras.optimizers import Adagrad
from keras.optimizers import Adamax
from keras.optimizers import Adadelta
# from tensorflow.keras.optimizers import AdamW
# from tensorflow.keras.optimizers import Adafactor

from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau

from keras.losses import MeanSquaredError
from keras.models import Sequential 
from keras.layers import Input

from utils.visualize import save_plot
from utils.metrics import score

from utils.general import convert_seconds
# import torch
# from torch.utils.data import DataLoader
import numpy as np
import pickle
from utils.general import yaml_load
from pathlib import Path
from utils.npy_utils import NpyFileAppend

# import torch.optim as optim
# import torch.nn as nn

from rich.progress import track
from rich.progress import Progress
from rich.progress import BarColumn 
from rich.progress import TextColumn
from rich.progress import TimeElapsedColumn
from rich.progress import MofNCompleteColumn
from rich.progress import TimeRemainingColumn

class BaseModel:
    def __init__(self, modelConfigs, save_dir='.'):
        self.history = None
        self.time_used = '0s'
        self.model = None
        self.modelConfigs = yaml_load(modelConfigs)

        self.dir_log          = 'logs'
        self.dir_plot         = 'plots'
        self.dir_value        = 'values'
        self.dir_model        = 'models'
        self.dir_weight       = 'weights'
        self.dir_architecture = 'architectures'
        self.mkdirs(path=save_dir)

    def mkdirs(self, path):
        path = Path(path)
        self.path_log          = path / self.dir_log
        self.path_plot         = path / self.dir_plot
        self.path_value        = path / self.dir_value
        self.path_model        = path / self.dir_model
        self.path_weight       = path / self.dir_weight
        self.path_architecture = path / self.dir_architecture

        for p in [self.path_log, self.path_plot, self.path_value, self.path_model, self.path_weight, self.path_architecture]: 
            p.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def build(self, *inputs):
        raise NotImplementedError 

    @abstractmethod
    def preprocessing(self, *inputs):
        raise NotImplementedError

    @abstractmethod
    def fit(self, *inputs):
        raise NotImplementedError

    @abstractmethod
    def save(self, *inputs):
        raise NotImplementedError

    @abstractmethod
    def load(self, *inputs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, *inputs):
        raise NotImplementedError

    def plot(self, save_dir, y, yhat, dataset):
        try:
            save_plot(filename=os.path.join(self.path_plot, f'{self.__class__.__name__}-{dataset}.png'),
                      data=[{'data': [range(len(y)), y],
                             'color': 'green',
                             'label': 'y'},
                            {'data': [range(len(yhat)), yhat],
                             'color': 'red',
                             'label': 'yhat'}],
                      xlabel='Sample',
                      ylabel='Value')
        except: pass

    def score(self, 
              y, 
              yhat, 
              # path=None,
              r):
        return score(y=y, 
                     yhat=yhat, 
                     # path=path, 
                     # model=self.__class__.__name__,
                     r=r)

    def ProgressBar(self):
        return Progress("[bright_cyan][progress.description]{task.description}",
                        BarColumn(),
                        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                        TextColumn("•Items"),
                        MofNCompleteColumn(), # "{task.completed}/{task.total}",
                        TextColumn("•Remaining"),
                        TimeRemainingColumn(),
                        TextColumn("•Total"),
                        TimeElapsedColumn())

class MachineLearningModel(BaseModel):
    def __init__(self, modelConfigs, save_dir, **kwargs):
        super().__init__(modelConfigs=modelConfigs, save_dir=save_dir)
        self.is_classifier = False
    
    def build(self): pass

    def preprocessing(self, x, classifier=False):
        # os.remove(temp_file)
        if classifier: res = [i.flatten().astype(int) for i in x]
        else: res = [i.flatten() for i in x]
        # else: res = np.ravel(x)
        # if classifier: res = x.flatten().astype(int)
        # else: res = x.flatten()
        return res

    # class DataGenerator:
    #     def __init__(self, x, y, batch_size, classifier=False):
    #         self.x = x
    #         self.y = y
    #         self.batch_size = batch_size
    #         self.classifier = classifier

    #     def __len__(self):
    #         return int(np.ceil(len(self.x) / float(self.batch_size)))

    #     def __getitem__(self, idx):
    #         batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
    #         batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

    #         processed_x = self.preprocessing(batch_x)
    #         processed_y = np.ravel(self.preprocessing(batch_y, classifier=self.classifier), order='C') 

    #         return processed_x, processed_y

    #     def preprocessing(self, data, classifier=False):
    #         processed_data = []
    #         for item in data:
    #             flattened_item = item.flatten()
    #             if classifier:
    #                 flattened_item = flattened_item.astype(int)
    #             processed_data.append(flattened_item)
    #         return np.array(processed_data)

    def fit(self, X_train, y_train, batchsz, time_as_int=False, **kwargs):
        start = time.time()
        # if self.is_classifier:
        #     y_train = np.ravel(vectorized(y_train, True), order='C') 
        # else:
        #     y_train = np.ravel(vectorized(y_train, False), order='C')

        temp_file = 'temp.npy'
        y_train = self.preprocessing(y_train, self.is_classifier)
        np.save(temp_file, y_train)
        y_train = np.load(temp_file, mmap_mode='r+')
        y_train = np.ravel(y_train, order='C') 
        x_train = self.preprocessing(X_train)
        

        # vectorized = np.vectorize(self.preprocessing)
        # y_train = np.ravel(vectorized(y_train, self.is_classifier), order='C')
        # x_train = vectorized(X_train)
        self.model.fit(X=x_train, 
                       y=y_train)
        self.time_used = time.time() - start
        if not time_as_int:
            self.time_used = convert_seconds(self.time_used)
        
    
    def save(self, file_name:str, extension:str='.pkl'):
        file_path = Path(self.path_weight, f'{file_name}{extension}')
        pickle.dump(self.model, open(Path(file_path).absolute(), "wb"))
        return file_path

    def load(self, weight):
        if not os.path.exists(weight): pass
        self.model = pickle.load(open(weight, "rb"))

    def predict(self, X, save=True, scaler=None):
        yhat = self.model.predict(self.preprocessing(x=X))
        if scaler is not None: 
            # yhat = yhat * (scaler['max'] - scaler['min']) +  scaler['min']
            yhat = yhat * scaler[0]['std'] - scaler[0]['mean']
            # print(yhat.shape)
            # yhat = yhat * (scaler[0]['max'] - scaler[0]['min']) +  scaler[0]['min']
            
        if save: 
            filename = self.path_value / f'yhat-{self.__class__.__name__}.npy'
            np.save(file=filename, 
                    arr=yhat, 
                    allow_pickle=True, 
                    fix_imports=True)
        return yhat

from tensorflow.keras.utils import Sequence 

class DataGenerator(Sequence):
    def __init__(self, x, y, batchsz):
        self.x, self.y = x, y
        self.batch_size = batchsz

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y

class TensorflowModel(BaseModel):
    def __init__(self, modelConfigs, input_shape, output_shape, save_dir, seed=941, **kwargs):
        super().__init__(modelConfigs=modelConfigs, save_dir=save_dir)
        self.function_dict = {
            'MSE'       : MeanSquaredError,
            'Adam'      : Adam,
            'SGD'       : lambda learning_rate: SGD(learning_rate=learning_rate, clipnorm=1.0),
            # 'AdamW'     : AdamW,
            'Nadam'     : Nadam,
            'RMSprop'   : RMSprop,
            # 'Adafactor' : Adafactor,
            'Adadelta'  : Adadelta,
            'Adagrad'   : Adagrad,
            'Adamax'    : Adamax,
            'Ftrl'      : Ftrl
        }
        self.units           = self.modelConfigs['units']
        self.activations     = [ele if ele != 'None' else None for ele in self.modelConfigs['activations']]
        self.dropouts        = self.modelConfigs['dropouts']
        self.seed            = seed
        self.input_shape     = input_shape
        self.output_shape    = output_shape
        
    def callbacks(self, patience, min_delta=0.001, extension='.h5'):
        if extension != '.h5':
            best = Path(self.path_weight, self.__class__.__name__, 'best')
            last = Path(self.path_weight, self.__class__.__name__, 'last')
        else:
            best = Path(self.path_weight, self.__class__.__name__)
            last = best
        best.mkdir(parents=True, exist_ok=True)
        last.mkdir(parents=True, exist_ok=True)

        return [EarlyStopping(monitor='val_loss', patience=patience, min_delta=min_delta), 
                ModelCheckpoint(filepath=os.path.join(best, f"{self.__class__.__name__}_best{extension}"),
                                save_best_only=True,
                                save_weights_only=True,
                                verbose=0), 
                ModelCheckpoint(filepath=os.path.join(last, f"{self.__class__.__name__}_last{extension}"),
                                save_best_only=False,
                                save_weights_only=True,
                                verbose=0),
                # ModelCheckpoint(filepath=os.path.join(model_path, f"{self.__class__.__name__}_best"),
                #                 save_best_only=True,
                #                 save_weights_only=False,
                #                 verbose=0), 
                # ModelCheckpoint(filepath=os.path.join(model_path, f"{self.__class__.__name__}_last"),
                #                 save_best_only=False,
                #                 save_weights_only=False,
                #                 verbose=0),
                ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.1,
                                  patience=patience / 5,
                                  verbose=0,
                                  mode='auto',
                                  min_delta=min_delta * 10,
                                  cooldown=0,
                                  min_lr=0), 
                CSVLogger(filename=os.path.join(self.path_log, f'{self.__class__.__name__}.csv'), separator=',', append=False)]  
        
    def preprocessing(self, x, y, batchsz):
        buffer_size = 512
        # if self.np_list:
        #     data = tf.data.Dataset.from_tensor_slices(list(zip(x, y)))\
        #                           .map(lambda item: tf.numpy_function(lambda item: np.load(file=item.decode(), mmap_mode='r'), 
        #                                                               [item], 
        #                                                               tf.float32),
        #                                num_parallel_calls=tf.data.AUTOTUNE)\
        #                           .shuffle(buffer_size=buffer_size, seed=self.seed, reshuffle_each_iteration=True)\
        #                           .batch(batchsz)\
        #                           .cache()\
        #                           .prefetch(buffer_size=tf.data.AUTOTUNE)
        # else:
        data = tf.data.Dataset.from_tensor_slices((x, y))\
                              .shuffle(buffer_size=buffer_size, seed=self.seed, reshuffle_each_iteration=True)\
                              .batch(batchsz)\
                              .cache()\
                              .prefetch(buffer_size=tf.data.AUTOTUNE)
        return data

    def build(self):
        try:
            self.model = Sequential(layers=None, name=self.__class__.__name__)
            # Input layer
            self.model.add(Input(shape=self.input_shape, name='Input_layer'))
            self.body()
            self.model.summary()
        except Exception as e:
            print(e)
            self.model = None

    def fit(self, X_train, y_train, X_val, y_val, patience, learning_rate, epochs, batchsz, optimizer='Adam', loss='MSE', time_as_int=False, **kwargs):
        start = time.time()
        self.model.compile(optimizer=self.function_dict[optimizer](learning_rate=learning_rate), loss=self.function_dict[loss]())
        # self.history = self.model.fit(self.preprocessing(x=X_train, y=y_train, batchsz=batchsz), 
        #                               validation_data=self.preprocessing(x=X_val, y=y_val, batchsz=batchsz),
        #                               epochs=epochs, 
        #                               callbacks=self.callbacks(patience=patience, min_delta=0.001))
        self.history = self.model.fit(DataGenerator(x=X_train, y=y_train, batchsz=batchsz), 
                                      validation_data=DataGenerator(x=X_val, y=y_val, batchsz=batchsz),
                                      epochs=epochs, 
                                      callbacks=self.callbacks(patience=patience, min_delta=0.001))
        self.time_used = time.time() - start
        if not time_as_int:
            self.time_used = convert_seconds(self.time_used)
        loss = self.history.history.get('loss')
        val_loss = self.history.history.get('val_loss')
        if all([len(loss)>1, len(val_loss)>1]):
            save_plot(filename=os.path.join(self.path_plot, f'{self.__class__.__name__}-Loss.png'),
                      data=[{'data': [range(len(loss)), loss],
                              'color': 'green',
                              'label': 'loss'},
                          {'data': [range(len(val_loss)), val_loss],
                              'color': 'red',
                              'label': 'val_loss'}],
                      xlabel='Epoch',
                      ylabel='Loss Value')

    def predict(self, X, save=True, scaler=None):
        yhat = self.model.predict(X, verbose=0)
        if scaler is not None: 
            # yhat = (yhat - scaler['min']) / (scaler['max'] - scaler['min'])
            yhat = yhat * scaler[0]['std'] - scaler[0]['mean']
        if save:
            filename = self.path_value / f'yhat-{self.__class__.__name__}.npy'
            np.save(file=filename, 
                    arr=yhat, 
                    allow_pickle=True, 
                    fix_imports=True)
        # # vectorized = np.vectorize(self.model.predict)
        # with self.ProgressBar() as progress:
        #     with NpyFileAppend(filename, delete_if_exists=True) as npfa:
        #         for x in progress.track(X, description='Predicting'):
        #             # print(X.shape, x.shape)
        #             npfa.append(self.model.predict(x[np.newaxis, :], verbose=0))
        # return np.load(file=filename, mmap_mode='r')
        return yhat
    
    def save(self, file_name:str, extension:str='.h5'):
        self.model.save_weights(Path(self.path_weight, file_name, f'{file_name}{extension}'))
        with open(os.path.join(self.path_architecture, f'{file_name}.json') , 'w') as outfile: json.dump(self.model.to_json(), outfile, indent=4)
        self.model.save(os.path.join(self.path_model, file_name))
        return os.path.join(self.path_weight, f'{file_name}{extension}')
    
    def load(self, weight):
        if os.path.exists(weight): self.model.load_weights(weight)


class LTSF_Linear_Base(TensorflowModel):
    def __init__(self, modelConfigs, input_shape, output_shape, save_dir, enc_in=1, seed=941, **kwargs):
        super().__init__(modelConfigs=modelConfigs, 
                         input_shape=input_shape, 
                         output_shape=output_shape,
                         save_dir=save_dir,
                         seed=seed)
        self.individual = self.modelConfigs['individual']
        self.enc_in = enc_in

    def save(self, file_name:str):
        file_path = os.path.join(self.path_weight, file_name, "ckpt")
        self.model.save_weights(Path(file_path).absolute())
        # with open(os.path.join(self.path_architecture, f'{file_name}.json') , 'w') as outfile: json.dump(self.model.to_json(), outfile, indent=4)
        return file_path

    def callbacks(self, patience, min_delta=0.001, extension=''):
        return super().callbacks(patience=patience, min_delta=min_delta, extension="")