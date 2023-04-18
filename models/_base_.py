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
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers import Adafactor

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

# import torch.optim as optim
# import torch.nn as nn

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
        self.path_log          = os.path.join(path, self.dir_log)
        self.path_plot         = os.path.join(path, self.dir_plot)
        self.path_value        = os.path.join(path, self.dir_value)
        self.path_model        = os.path.join(path, self.dir_model)
        self.path_weight       = os.path.join(path, self.dir_weight)
        self.path_architecture = os.path.join(path, self.dir_architecture)

        for p in [self.path_log, self.path_plot, self.path_value, self.path_model, self.path_weight, self.path_architecture]: 
            os.makedirs(name=p, exist_ok=True)

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

    def score(self, y, yhat, r, path=None):
        return score(y=y, yhat=yhat, r=r, path=path, model=self.__class__.__name__)

class MachineLearningModel(BaseModel):
    def __init__(self, modelConfigs, save_dir, **kwargs):
        super().__init__(modelConfigs=modelConfigs, save_dir=save_dir)
        self.is_classifier = False
    
    def build(self): pass

    def preprocessing(self, x):
        return [i.flatten() for i in x]

    def fit(self, X_train, y_train, **kwargs):
        start = time.time()
        if self.is_classifier:
            y_train = np.ravel([i.astype(int) for i in self.preprocessing(x=y_train)], order='C') 
        else:
            y_train = np.ravel(self.preprocessing(x=y_train), order='C')
        self.model.fit(X=self.preprocessing(x=X_train), 
                       y=y_train)
        self.time_used = convert_seconds(time.time() - start)
    
    def save(self, file_name:str, extension:str='.pkl'):
        file_path = os.path.join(self.path_weight, file_name+extension)
        pickle.dump(self.model, open(Path(file_path).absolute(), "wb"))
        return file_path

    def load(self, weight):
        if not os.path.exists(weight): pass
        self.model = pickle.load(open(weight, "rb"))

    def predict(self, X):
        return self.model.predict(self.preprocessing(x=X)) 

class TensorflowModel(BaseModel):
    def __init__(self, modelConfigs, input_shape, output_shape, save_dir, normalize_layer=None, seed=941, **kwargs):
        super().__init__(modelConfigs=modelConfigs, save_dir=save_dir)
        self.function_dict = {
            'MSE'       : MeanSquaredError,
            'Adam'      : Adam,
            'SGD'       : SGD,
            'AdamW'     : AdamW,
            'Nadam'     : Nadam,
            'RMSprop'   : RMSprop,
            'Adafactor' : Adafactor,
            'Adadelta'  : Adadelta,
            'Adagrad'   : Adagrad,
            'Adamax'    : Adamax,
            'Ftrl'      : Ftrl
        }
        self.units           = self.modelConfigs['units']
        self.activations     = [ele if ele != 'None' else None for ele in self.modelConfigs['activations']]
        self.dropouts        = self.modelConfigs['dropouts']
        self.seed            = seed
        self.normalize_layer = normalize_layer
        self.input_shape     = input_shape
        self.output_shape    = output_shape
        
    def callbacks(self, patience, min_delta=0.001, extension='.h5'):
        return [EarlyStopping(monitor='val_loss', patience=patience, min_delta=min_delta), 
                ModelCheckpoint(filepath=os.path.join(self.path_weight, f"{self.__class__.__name__}_best{extension}"),
                                save_best_only=True,
                                save_weights_only=True,
                                verbose=0), 
                ModelCheckpoint(filepath=os.path.join(self.path_weight, f"{self.__class__.__name__}_last{extension}"),
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
        return tf.data.Dataset.from_tensor_slices((x, y)).shuffle(buffer_size=5120, seed=self.seed, reshuffle_each_iteration=True).batch(batchsz).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

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

    def fit(self, X_train, y_train, X_val, y_val, patience, learning_rate, epochs, batchsz, optimizer='Adam', loss='MSE', **kwargs):
        start = time.time()
        self.model.compile(optimizer=self.function_dict[optimizer](learning_rate=learning_rate), loss=self.function_dict[loss]())
        self.history = self.model.fit(self.preprocessing(x=X_train, y=y_train, batchsz=batchsz), 
                                      validation_data=self.preprocessing(x=X_val, y=y_val, batchsz=batchsz),
                                      epochs=epochs, 
                                      callbacks=self.callbacks(patience=patience, min_delta=0.001))
        self.time_used = convert_seconds(time.time() - start)
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

    def predict(self, X):
        return self.model.predict(X, verbose=0)
    
    def save(self, file_name:str, extension:str='.h5'):
        self.model.save_weights(os.path.join(self.path_weight, f'{file_name}{extension}'))
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
                         normalize_layer=None,
                         seed=seed)
        self.individual = self.modelConfigs['individual']
        self.enc_in = enc_in

    def save(self, file_name:str):
        file_path = os.path.join(self.path_weight, file_name, "ckpt")
        self.model.save_weights(Path(file_path).absolute())
        return file_path

    def callbacks(self, patience, min_delta=0.001, extension=''):
        return super().callbacks(patience=patience, min_delta=min_delta, extension="ckpt")