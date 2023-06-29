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
import numpy as np
from keras import layers, models, initializers
from keras_self_attention import SeqSelfAttention
from keras_multi_head import MultiHeadAttention

import tensorflow as tf


# from utils.activations import SoftRootSign

class VanillaLSTM__Tensorflow(TensorflowModel):
    def body(self):
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

class EmbeddedLSTM__Tensorflow(TensorflowModel):
    def build(self):
        num_of_ID = 10_000
        input_len = self.input_shape[0]
        embedding_dim_size = 128
        # self.input_shape = (i, c)
        inputs = layers.Input(shape=self.input_shape)

        input_series_id = inputs[..., 0:1]
        input_encoded_time = inputs[..., 1:7]
        input_series = inputs[..., 7:]

        series_id_embedding = layers.Embedding(input_dim=num_of_ID + 1, 
                                                    output_dim=embedding_dim_size, 
                                                    name='series_id_embedding_layer',
                                                    embeddings_initializer=initializers.RandomUniform(seed=self.seed))(input_series_id[:,0:1,:])
        
        series_id_embedded = layers.Flatten()(series_id_embedding)
        features_combined = layers.concatenate([layers.RepeatVector(input_len)(series_id_embedded), input_encoded_time, input_series])

        x = LSTM(name='LSTM_layer',
                 units=self.units[0],
                 return_sequences=False,
                 kernel_initializer=GlorotUniform(seed=self.seed), 
                 activation=self.activations[0])(features_combined)
        
        dense1_input = layers.concatenate([series_id_embedded, x])
        
        # FC Layer
        x = Dense(name='Fully_Connected_layer',
                             units=self.units[1],
                             kernel_initializer=GlorotUniform(seed=self.seed),
                             activation=self.activations[1])(dense1_input)
        # Output Layer
        outputs = Dense(name='Output_layer',
                             units=self.output_shape, 
                             kernel_initializer=GlorotUniform(seed=self.seed),
                             activation=self.activations[2])(x)
    
        
        self.model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

        self.model.summary()

class EmbeddedBiLSTM__Tensorflow(TensorflowModel):
    def build(self):
        num_of_ID = 10_000
        input_len = self.input_shape[0]
        embedding_dim_size = 128
        # self.input_shape = (i, c)
        inputs = layers.Input(shape=self.input_shape)

        input_series_id = inputs[..., 0:1]
        input_encoded_time = inputs[..., 1:7]
        input_series = inputs[..., 7:]

        series_id_embedding = layers.Embedding(input_dim=num_of_ID + 1, 
                                                    output_dim=embedding_dim_size, 
                                                    name='series_id_embedding_layer',
                                                    embeddings_initializer=initializers.RandomUniform(seed=self.seed))(input_series_id[:,0:1,:])
        
        series_id_embedded = layers.Flatten()(series_id_embedding)
        features_combined = layers.concatenate([layers.RepeatVector(input_len)(series_id_embedded), input_encoded_time, input_series])

        x = Bidirectional(LSTM(units=self.units[0],
                 return_sequences=True,
                 kernel_initializer=GlorotUniform(seed=self.seed), 
                 activation=self.activations[0]))(features_combined)
        
        x = Bidirectional(LSTM(units=self.units[0],
                 return_sequences=False,
                 kernel_initializer=GlorotUniform(seed=self.seed), 
                 activation=self.activations[0]))(x)
        
        dense1_input = layers.concatenate([series_id_embedded, x])
        
        # FC Layer
        x = Dense(name='Fully_Connected_layer',
                             units=self.units[1],
                             kernel_initializer=GlorotUniform(seed=self.seed),
                             activation=self.activations[1])(dense1_input)
        # Output Layer
        outputs = Dense(name='Output_layer',
                             units=self.output_shape, 
                             kernel_initializer=GlorotUniform(seed=self.seed),
                             activation=self.activations[2])(x)
    
        
        self.model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

        self.model.summary()

class BiLSTM__Tensorflow(TensorflowModel):
    def body(self):
        # BiLSTM Layer 1 
        self.model.add(Bidirectional(name='BiLSTM_layer_1',
                                     layer=LSTM(units=self.units[0],
                                                return_sequences=True,
                                                kernel_initializer=GlorotUniform(seed=self.seed),
                                                activation=self.activations[0])))
        # BiLSTM Layer 2 
        self.model.add(Bidirectional(name='BiLSTM_layer_2',
                                     layer=LSTM(units=self.units[1],
                                                return_sequences=False,
                                                kernel_initializer=GlorotUniform(seed=self.seed),
                                                activation=self.activations[1])))
        # FC Layer
        self.model.add(Dense(name='Fully_Connected_layer',
                             units=self.units[2],
                             kernel_initializer=GlorotUniform(seed=self.seed),
                             activation=self.activations[2]))

        # self.model.add(SoftRootSign(trainable=True))

        # Output Layer
        self.model.add(Dense(name='Output_layer',
                             units=self.output_shape, 
                             kernel_initializer=GlorotUniform(seed=self.seed),
                             activation=self.activations[3]))

class Time2Vec(layers.Layer):
    def __init__(self, kernel_size):
        super(Time2Vec, self).__init__(trainable=True)     
        self.k = kernel_size
    
    def build(self, input_shape):
        self.wb = self.add_weight(
            shape=(input_shape[1], 1),
            initializer='uniform',
            trainable=True,
            name='wb_weight'
        )
        
        self.bb = self.add_weight(
            shape=(input_shape[1], 1),
            initializer='uniform',
            trainable=True,
            name='bb_weight'
        )
        
        self.wa = self.add_weight(
            shape=(input_shape[-1], self.k),
            initializer='uniform',
            trainable=True,
            name='wa_weight'
        )
        
        self.ba = self.add_weight(
            shape=(input_shape[1], self.k),
            initializer='uniform',
            trainable=True,
            name='ba_weight'
        )
        
        super(Time2Vec, self).build(input_shape)
    
    def call(self, inputs):
        bias = self.wb * inputs + self.bb
        wgts = tf.math.sin( tf.matmul(inputs, self.wa) + self.ba)
        return layers.Concatenate(axis=-1)([wgts, bias])
    
    def get_config(self):
        config = super(Time2Vec, self).get_config()
        config.update({'kernel_size': self.k})
        return config
    
class Time2Vec_BiLSTM__Tensorflow(TensorflowModel):
    def build(self):
        input_series_value = layers.Input(shape=self.input_shape, name='input_series_value')
        time2vec_series_value = Time2Vec(kernel_size=self.units[0])(input_series_value)
        
        x = layers.Bidirectional(name='BiLSTM_layer_1',
                                layer=layers.LSTM(units=self.units[1],
                                        return_sequences=True,
                                        kernel_initializer=initializers.GlorotUniform(seed=self.seed),
                                        activation=self.activations[1]))(time2vec_series_value)
        x = layers.Bidirectional(name='BiLSTM_layer_2',
                                layer=layers.LSTM(units=self.units[2],
                                        return_sequences=False,
                                        kernel_initializer=initializers.GlorotUniform(seed=self.seed),
                                        activation=self.activations[2]))(x)               

        x = layers.Dense(name='Fully_Connected_layer_2',
                        units=self.units[3],
                        kernel_initializer=initializers.GlorotUniform(seed=self.seed),
                        activation=self.activations[3])(x)

        outputs = layers.Dense(name='Output_layer',
                        units=self.output_shape, 
                        kernel_initializer=initializers.GlorotUniform(seed=self.seed),
                        activation=self.activations[4])(x)

        self.model = models.Model(inputs=[input_series_value], outputs=[outputs])
        self.model.summary()
    
class EmbeddedTime2Vec_BiLSTM__Tensorflow(TensorflowModel):
    def build(self):
        num_of_ID = 10_000
        input_len = self.input_shape[0]
        embedding_dim_size = 128
        # self.input_shape = (i, c)
        inputs = layers.Input(shape=self.input_shape)

        input_series_id = inputs[..., 0:1]
        input_encoded_time = inputs[..., 1:7]
        input_series = inputs[..., 7:]

        series_id_embedding = layers.Embedding(input_dim=num_of_ID + 1, 
                                                    output_dim=embedding_dim_size, 
                                                    name='series_id_embedding_layer',
                                                    embeddings_initializer=initializers.RandomUniform(seed=self.seed))(input_series_id[:,0:1,:])
        
        series_id_embedded = layers.Flatten()(series_id_embedding)
        time2vec_series_value = Time2Vec(kernel_size=self.units[0])(input_series)

        features_combined = layers.concatenate([layers.RepeatVector(input_len)(series_id_embedded),  input_encoded_time, time2vec_series_value])

        
        x = layers.Bidirectional(name='BiLSTM_layer_1',
                                layer=layers.LSTM(units=self.units[1],
                                        return_sequences=True,
                                        kernel_initializer=initializers.GlorotUniform(seed=self.seed),
                                        activation=self.activations[1]))(time2vec_series_value)
        x = layers.Bidirectional(name='BiLSTM_layer_2',
                                layer=layers.LSTM(units=self.units[2],
                                        return_sequences=False,
                                        kernel_initializer=initializers.GlorotUniform(seed=self.seed),
                                        activation=self.activations[2]))(x) 
        
        dense1_input = layers.concatenate([series_id_embedded, x])              

        x = layers.Dense(name='Fully_Connected_layer_2',
                        units=self.units[3],
                        kernel_initializer=initializers.GlorotUniform(seed=self.seed),
                        activation=self.activations[3])(dense1_input)

        outputs = layers.Dense(name='Output_layer',
                        units=self.output_shape, 
                        kernel_initializer=initializers.GlorotUniform(seed=self.seed),
                        activation=self.activations[4])(x)

        self.model = models.Model(inputs=[inputs], outputs=[outputs])
        self.model.summary()




# def build_simple_Time2Vec_Bi_LSTSM(seed, time2vec_ouputs_size, input_len, predict_len, num_of_series):

    
#     return model

class SelfAttention_BiLSTSM__Tensorflow(TensorflowModel):
    def build(self):
        input_series_value = layers.Input(shape=self.input_shape, name='input_series_value')

        x = layers.Bidirectional(name='BiLSTM_layer_1',
                                layer=layers.LSTM(units=self.units[0],
                                        return_sequences=True,
                                        kernel_initializer=initializers.GlorotUniform(seed=self.seed),
                                        activation=self.activations[0]))(input_series_value)

        x = SeqSelfAttention(attention_activation = 'linear')(x)

        x = layers.Bidirectional(name='BiLSTM_layer_2',
                                 layer=layers.LSTM(units=self.units[1],
                                        return_sequences=False,
                                        kernel_initializer=initializers.GlorotUniform(seed=self.seed),
                                        activation=self.activations[1]))(x)               

        x = layers.Dense(name='Fully_Connected_layer_2',
                        units=self.units[2],
                        kernel_initializer=initializers.GlorotUniform(seed=self.seed),
                        activation=self.activations[2])(x)

        outputs = layers.Dense(name='Output_layer',
                        units=self.output_shape, 
                        kernel_initializer=initializers.GlorotUniform(seed=self.seed),
                        activation=self.activations[3])(x)

        self.model = models.Model(inputs=[input_series_value], outputs=[outputs])
        self.model.summary()
        
class EmbeddedSelfAttention_BiLSTSM__Tensorflow(TensorflowModel):
    def build(self):
        num_of_ID = 10_000
        input_len = self.input_shape[0]
        embedding_dim_size = 128
        # self.input_shape = (i, c)
        inputs = layers.Input(shape=self.input_shape)

        input_series_id = inputs[..., 0:1]
        input_encoded_time = inputs[..., 1:7]
        input_series = inputs[..., 7:]

        series_id_embedding = layers.Embedding(input_dim=num_of_ID + 1, 
                                                    output_dim=embedding_dim_size, 
                                                    name='series_id_embedding_layer',
                                                    embeddings_initializer=initializers.RandomUniform(seed=self.seed))(input_series_id[:,0:1,:])
        
        series_id_embedded = layers.Flatten()(series_id_embedding)
        features_combined = layers.concatenate([layers.RepeatVector(input_len)(series_id_embedded), input_encoded_time, input_series])

        x = layers.Bidirectional(name='BiLSTM_layer_1',
                                layer=layers.LSTM(units=self.units[0],
                                        return_sequences=True,
                                        kernel_initializer=initializers.GlorotUniform(seed=self.seed),
                                        activation=self.activations[0]))(features_combined)

        x = SeqSelfAttention(attention_activation = 'linear')(x)

        x = layers.Bidirectional(name='BiLSTM_layer_2',
                                 layer=layers.LSTM(units=self.units[1],
                                        return_sequences=False,
                                        kernel_initializer=initializers.GlorotUniform(seed=self.seed),
                                        activation=self.activations[1]))(x)               

        dense1_input = layers.concatenate([series_id_embedded, x]) 
        x = layers.Dense(name='Fully_Connected_layer_2',
                        units=self.units[2],
                        kernel_initializer=initializers.GlorotUniform(seed=self.seed),
                        activation=self.activations[2])(dense1_input)

        outputs = layers.Dense(name='Output_layer',
                        units=self.output_shape, 
                        kernel_initializer=initializers.GlorotUniform(seed=self.seed),
                        activation=self.activations[3])(x)

        self.model = models.Model(inputs=[inputs], outputs=[outputs])
        self.model.summary()

      
class Multihead_BiLSTSM__Tensorflow(TensorflowModel):
    def build(self):
        input_series_value = layers.Input(shape=self.input_shape, name='input_series_value')

        
        x = layers.Bidirectional(name='BiLSTM_layer_1',
                                layer=layers.LSTM(units=self.units[0],
                                        return_sequences=True,
                                        kernel_initializer=initializers.GlorotUniform(seed=self.seed),
                                        activation=self.activations[0]))(input_series_value)

        x = MultiHeadAttention(head_num = x.shape[-1])(x)

        x = layers.Bidirectional(name='BiLSTM_layer_2',
                                layer=layers.LSTM(units=self.units[1],
                                        return_sequences=False,
                                        kernel_initializer=initializers.GlorotUniform(seed=self.seed),
                                        activation=self.activations[1]))(x)               

        x = layers.Dense(name='Fully_Connected_layer_2',
                        units=self.units[2],
                        kernel_initializer=initializers.GlorotUniform(seed=self.seed),
                        activation=self.activations[2])(x)

        outputs = layers.Dense(name='Output_layer',
                        units=self.output_shape, 
                        kernel_initializer=initializers.GlorotUniform(seed=self.seed),
                        activation=self.activations[3])(x)

        self.model = models.Model(inputs=[input_series_value], outputs=[outputs])
        self.model.summary()



class EmbeddedMultihead_BiLSTSM__Tensorflow(TensorflowModel):
    def build(self):
        num_of_ID = 10_000
        input_len = self.input_shape[0]
        embedding_dim_size = 128
        # self.input_shape = (i, c)
        inputs = layers.Input(shape=self.input_shape)

        input_series_id = inputs[..., 0:1]
        input_encoded_time = inputs[..., 1:7]
        input_series = inputs[..., 7:]

        series_id_embedding = layers.Embedding(input_dim=num_of_ID + 1, 
                                                    output_dim=embedding_dim_size, 
                                                    name='series_id_embedding_layer',
                                                    embeddings_initializer=initializers.RandomUniform(seed=self.seed))(input_series_id[:,0:1,:])
        
        series_id_embedded = layers.Flatten()(series_id_embedding)
        features_combined = layers.concatenate([layers.RepeatVector(input_len)(series_id_embedded), input_encoded_time, input_series])


        
        x = layers.Bidirectional(name='BiLSTM_layer_1',
                                layer=layers.LSTM(units=self.units[0],
                                        return_sequences=True,
                                        kernel_initializer=initializers.GlorotUniform(seed=self.seed),
                                        activation=self.activations[0]))(features_combined)

        x = MultiHeadAttention(head_num = x.shape[-1])(x)

        x = layers.Bidirectional(name='BiLSTM_layer_2',
                                layer=layers.LSTM(units=self.units[1],
                                        return_sequences=False,
                                        kernel_initializer=initializers.GlorotUniform(seed=self.seed),
                                        activation=self.activations[1]))(x)               

        dense1_input = layers.concatenate([series_id_embedded, x]) 
        x = layers.Dense(name='Fully_Connected_layer_2',
                        units=self.units[2],
                        kernel_initializer=initializers.GlorotUniform(seed=self.seed),
                        activation=self.activations[2])(dense1_input)

        outputs = layers.Dense(name='Output_layer',
                        units=self.output_shape, 
                        kernel_initializer=initializers.GlorotUniform(seed=self.seed),
                        activation=self.activations[3])(x)

        self.model = models.Model(inputs=[inputs], outputs=[outputs])
        self.model.summary()