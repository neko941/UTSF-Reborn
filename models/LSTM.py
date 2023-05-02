import numpy as np

from models._base_ import TensorflowModel
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Conv3D
from keras.layers import Flatten
from keras.layers import ConvLSTM2D
from keras.layers import Bidirectional
from keras.layers import BatchNormalization
from keras.initializers import GlorotUniform

# from utils.activations import SoftRootSign

class VanillaLSTM__Tensorflow(TensorflowModel):
    def body(self):
        # Normalization
        if self.normalize_layer: self.model.add(self.normalize_layer)
        # LSTM Layer 
        self.model.add(LSTM(name='LSTM_layer',
                            units=self.units[0],
                            return_sequences=False,
                            kernel_initializer=GlorotUniform(seed=self.seed), 
                            activation=self.activations[0]))
        # FC Layer
        self.model.add(Dense(name='Fully_Connected_layer',
                             units=self.units[1],
                             kernel_initializer=GlorotUniform(seed=self.seed),
                             activation=self.activations[1]))
        # Output Layer
        self.model.add(Dense(name='Output_layer',
                             units=self.output_shape, 
                             kernel_initializer=GlorotUniform(seed=self.seed),
                             activation=self.activations[2]))

class BiLSTM__Tensorflow(TensorflowModel):
    def body(self):
        # Normalization
        if self.normalize_layer: self.model.add(self.normalize_layer)
        # BiLSTM Layer 1 
        self.model.add(Bidirectional(name='BiLSTM_layer_1',
                                     layer=LSTM(units=self.units[0],
                                                return_sequences=True,
                                                kernel_initializer=GlorotUniform(seed=self.seed),
                                                activation=self.activations[0])))
        # BiLSTM Layer 2 
        self.model.add(Bidirectional(name='BiLSTM_layer_2',
                                     layer=LSTM(units=self.units[1],
                                                return_sequences=True,
                                                kernel_initializer=GlorotUniform(seed=self.seed),
                                                activation=self.activations[1])))
        # BiLSTM Layer 3 
        self.model.add(Bidirectional(name='BiLSTM_layer_3',
                                     layer=LSTM(units=self.units[2],
                                                return_sequences=False,
                                                kernel_initializer=GlorotUniform(seed=self.seed),
                                                activation=self.activations[2])))
        # FC Layer
        self.model.add(Dense(name='Fully_Connected_layer',
                             units=self.units[3],
                             kernel_initializer=GlorotUniform(seed=self.seed),
                             activation=self.activations[3]))

        # self.model.add(SoftRootSign(trainable=True))

        # Output Layer
        self.model.add(Dense(name='Output_layer',
                             units=self.output_shape, 
                             kernel_initializer=GlorotUniform(seed=self.seed),
                             activation=self.activations[4]))
