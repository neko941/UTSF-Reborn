import os
from abc import abstractmethod
import json
import time

import tensorflow as tf

from keras.optimizers import SGD
from keras.optimizers import Adam
from tensorflow.keras.optimizers import AdamW
from keras.losses import MeanSquaredError
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
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
    def __init__(self):
        self.history = None
        self.dir_weight = 'weights'
        self.dir_value = 'values'
        self.dir_log = 'logs'
        self.dir_model = 'models'
        self.dir_architecture = 'architectures'

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
        if self.output_shape == 1:
            visualize_path = os.path.join(save_dir, 'plots')
            os.makedirs(name=visualize_path, exist_ok=True)

            save_plot(filename=os.path.join(visualize_path, f'{self.__class__.__name__}-{dataset}.png'),
                      data=[{'data': [range(len(y)), y],
                             'color': 'green',
                             'label': 'y'},
                            {'data': [range(len(yhat)), yhat],
                             'color': 'red',
                             'label': 'yhat'}],
                      xlabel='Sample',
                      ylabel='Value')

    def score(self, y, yhat, r, path=None):
        return score(y=y, yhat=yhat, r=r, path=path, model=self.__class__.__name__)

class TensorflowModel(BaseModel):
    def __init__(self, modelConfigs, input_shape, output_shape, normalize_layer=None, seed=941, **kwargs):
        super().__init__()
        self.function_dict = {'Adam' : Adam,
                              'MSE' : MeanSquaredError,
                              'SGD' : SGD,
                              'AdamW' : AdamW}
        self.modelConfigs = yaml_load(modelConfigs)
        self.units = self.modelConfigs['units']
        self.activations = [ele if ele != 'None' else None for ele in self.modelConfigs['activations']]
        self.dropouts = self.modelConfigs['dropouts']
        self.seed = seed
        self.normalize_layer = normalize_layer
        self.input_shape = input_shape
        self.output_shape = output_shape
        
    def callbacks(self, patience, save_dir, min_delta=0.001, extension='.h5'):
        weight_path = os.path.join(save_dir, 'weights')
        os.makedirs(name=weight_path, exist_ok=True)
        log_path = os.path.join(save_dir, 'logs')
        os.makedirs(name=log_path, exist_ok=True)

        return [EarlyStopping(monitor='val_loss', patience=patience, min_delta=min_delta), 
                ModelCheckpoint(filepath=os.path.join(weight_path, f"{self.__class__.__name__}_best{extension}"),
                                save_best_only=True,
                                save_weights_only=False,
                                verbose=0), 
                ModelCheckpoint(filepath=os.path.join(weight_path, f"{self.__class__.__name__}_last{extension}"),
                                save_best_only=False,
                                save_weights_only=False,
                                verbose=0),
                ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.1,
                                  patience=patience / 5,
                                  verbose=0,
                                  mode='auto',
                                  min_delta=min_delta * 10,
                                  cooldown=0,
                                  min_lr=0), 
                CSVLogger(filename=os.path.join(log_path, f'{self.__class__.__name__}.csv'), separator=',', append=False)]  
        
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

    def fit(self, X_train, y_train, X_val, y_val, patience, learning_rate, epochs, save_dir, batchsz, optimizer='Adam', loss='MSE', **kwargs):
        start = time.time()
        self.model.compile(optimizer=self.function_dict[optimizer](learning_rate=learning_rate), loss=self.function_dict[loss]())
        self.history = self.model.fit(self.preprocessing(x=X_train, y=y_train, batchsz=batchsz), 
                                      validation_data=self.preprocessing(x=X_val, y=y_val, batchsz=batchsz),
                                      epochs=epochs, 
                                      callbacks=self.callbacks(patience=patience, save_dir=save_dir, min_delta=0.001))
        self.time_used = convert_seconds(time.time() - start)
        loss = self.history.history.get('loss')
        val_loss = self.history.history.get('val_loss')
        if all([len(loss)>1, len(val_loss)>1]):
            os.makedirs(os.path.join(save_dir, 'plots'), exist_ok=True)
            save_plot(filename=os.path.join(save_dir, 'plots', f'{self.__class__.__name__}-Loss.png'),
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
    
    def save(self, file_name:str, save_dir:str='.', extension:str='.h5'):
        os.makedirs(name=os.path.join(save_dir, 'weights'), exist_ok=True)
        os.makedirs(name=os.path.join(save_dir, 'architectures'), exist_ok=True)
        os.makedirs(name=os.path.join(save_dir, 'models'), exist_ok=True)
        
        weight_path = os.path.join(save_dir, 'weights', f'{file_name}.h5')
        architecture_path = os.path.join(save_dir, 'architectures', f'{file_name}.json') 
        model_path = os.path.join(save_dir, 'models', file_name)
        
        self.model.save_weights(weight_path)
        with open(architecture_path, 'w') as outfile: json.dump(self.model.to_json(), outfile, indent=4)
        self.model.save(model_path)
        return weight_path

        # pickle.dump(self.model, open(Path(file_path).absolute(), "wb"))
        # return file_path
    
    def load(self, weight):
        if os.path.exists(weight): self.model.load_weights(weight)
