import tensorflow as tf
from keras.layers import GRU
from keras.layers import Dense
from keras.layers import Bidirectional
from keras.initializers import GlorotUniform

from models._base_ import TensorflowModel

class VanillaGRU__Tensorflow(TensorflowModel):
    def body(self):
        # GRU Layer 1
        self.model.add(GRU(name='GRU_layer',
                           units=self.units[0],
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

class BiGRU__Tensorflow(TensorflowModel):
    def body(self):
        # BiGRU Layer 1 
        self.model.add(Bidirectional(name='BiGRU_layer_1',
                                     layer=GRU(units=self.units[0],
                                               return_sequences=True,
                                               kernel_initializer=GlorotUniform(seed=self.seed),
                                               activation=self.activations[0])))
        # BiGRU Layer 2 
        self.model.add(Bidirectional(name='BiGRU_layer_2',
                                     layer=GRU(units=self.units[1],
                                               return_sequences=True,
                                               kernel_initializer=GlorotUniform(seed=self.seed),
                                               activation=self.activations[1])))
        # BiGRU Layer 3 
        self.model.add(Bidirectional(name='BiGRU_layer_3',
                                     layer=GRU(units=self.units[2],
                                               return_sequences=False,
                                               kernel_initializer=GlorotUniform(seed=self.seed),
                                               activation=self.activations[2])))
        # FC Layer
        self.model.add(Dense(name='Fully_Connected_layer',
                             units=self.units[3],
                             kernel_initializer=GlorotUniform(seed=self.seed),
                             activation=self.activations[3]))

        # Output Layer
        self.model.add(Dense(name='Output_layer',
                             units=self.output_shape, 
                             kernel_initializer=GlorotUniform(seed=self.seed),
                             activation=self.activations[4]))