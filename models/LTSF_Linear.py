import os
import tensorflow as tf
from pathlib import Path
from keras.layers import Dense
from models._base_ import LTSF_Linear_Base
from keras.layers import AveragePooling1D
from keras.layers import Layer
from keras import layers
class MovingAvg__Tensorflow(Layer):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(MovingAvg__Tensorflow, self).__init__()
        self.kernel_size = kernel_size
        self.avg = AveragePooling1D(pool_size=kernel_size, strides=stride, padding='valid')

    def call(self, x):
        # padding on the both ends of time series
        front = tf.tile(x[:, 0:1, :], multiples=[1, (self.kernel_size - 1) // 2, 1])
        end = tf.tile(x[:, -1:, :], multiples=[1, (self.kernel_size - 1) // 2, 1])
        x = tf.concat([front, x, end], axis=1)
        x = self.avg(x)
        return x


class SeriesDecomp__Tensorflow(Layer):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(SeriesDecomp__Tensorflow, self).__init__()
        self.moving_avg = MovingAvg__Tensorflow(kernel_size, stride=1)

    def call(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinear__Tensorflow(tf.keras.Model):
    """
    Decomposition-Linear
    """
    def __init__(self, seq_len, pred_len, enc_in, individual):
        super(DLinear__Tensorflow, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
        self.individual = individual

        # Decompsition Kernel Size
        kernel_size = 25
        self.decomposition = SeriesDecomp__Tensorflow(kernel_size)
        

        if self.individual:
            self.Linear_Seasonal = []
            self.Linear_Trend = []
            for i in range(self.channels):
                self.Linear_Seasonal.append(Dense(self.pred_len))
                self.Linear_Trend.append(Dense(self.pred_len))
                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].kernel = tf.Variable((1/self.seq_len)*tf.ones([self.seq_len, self.pred_len]))
                # self.Linear_Trend[i].kernel = tf.Variable((1/self.seq_len)*tf.ones([self.seq_len, self.pred_len]))
        else:
            self.Linear_Seasonal = Dense(self.pred_len)
            self.Linear_Trend = Dense(self.pred_len)
            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.kernel = tf.Variable((1/self.seq_len)*tf.ones([self.seq_len, self.pred_len]))
            # self.Linear_Trend.kernel = tf.Variable((1/self.seq_len)*tf.ones([self.seq_len, self.pred_len]))
        self.final_layer = Dense(self.pred_len)

    def call(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decomposition(x)

        if self.individual:
            seasonal_output = tf.concat([tf.expand_dims(self.Linear_Seasonal[i](x[:,:,i]), axis=-1) for i in range(self.channels)], axis=-1)
            trend_output = tf.concat([tf.expand_dims(self.Linear_Trend[i](x[:,:,i]), axis=-1) for i in range(self.channels)], axis=-1)
            x = seasonal_output + trend_output
        else:
            seasonal_init, trend_init = tf.transpose(seasonal_init, perm=[0,2,1]), tf.transpose(trend_init, perm=[0,2,1])
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
            x = seasonal_output + trend_output
            x = tf.transpose(x, perm=[0,2,1]) # to [Batch, Output length, Channel]

        # print(x.shape)
        if self.pred_len==1: x = tf.squeeze(self.final_layer(x), axis=-1)
        return x # [Batch, Output length, Channel]


class  Embedded_DLinear__Tensorflow(tf.keras.Model):
    """
    Decomposition-Linear
    """
    def __init__(self, seq_len, pred_len, enc_in, individual, seed):
        super(Embedded_DLinear__Tensorflow, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
        self.individual = individual

        # Decompsition Kernel Size
        kernel_size = 25
        self.decomposition = SeriesDecomp__Tensorflow(kernel_size)

        num_of_ID = 10_000
        input_len = seq_len[0]
        embedding_dim_size = 128
        # self.input_shape = (i, c)
        self.embedding_layer = layers.Embedding(input_dim=num_of_ID + 1, 
                                                    output_dim=embedding_dim_size, 
                                                    name='series_id_embedding_layer',
                                                    embeddings_initializer=initializers.RandomUniform(seed=seed))

        self.flat1 = layers.Flatten()
        self.repeat = layers.RepeatVector(input_len)
        
        if self.individual:
            self.Linear_Seasonal = []
            self.Linear_Trend = []
            for i in range(self.channels):
                self.Linear_Seasonal.append(Dense(self.pred_len))
                self.Linear_Trend.append(Dense(self.pred_len))
                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].kernel = tf.Variable((1/self.seq_len)*tf.ones([self.seq_len, self.pred_len]))
                # self.Linear_Trend[i].kernel = tf.Variable((1/self.seq_len)*tf.ones([self.seq_len, self.pred_len]))
        else:
            self.Linear_Seasonal = Dense(self.pred_len)
            self.Linear_Trend = Dense(self.pred_len)
            self.Others_Dense = Dense(self.pred_len)
            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.kernel = tf.Variable((1/self.seq_len)*tf.ones([self.seq_len, self.pred_len]))
            # self.Linear_Trend.kernel = tf.Variable((1/self.seq_len)*tf.ones([self.seq_len, self.pred_len]))
        self.final_layer = Dense(self.pred_len)

    def call(self, inputs):
        input_series_id = inputs[..., 0:1]
        input_encoded_time = inputs[..., 1:7]
        input_series = inputs[..., 7:]

        series_id_embedding = self.embedding_layer(input_series_id[:, 0:1,:])
        
        series_id_embedded = self.flat1(series_id_embedding)
        
        
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decomposition(input_series)
        features_combined = layers.concatenate([self.repeat(series_id_embedded),  input_encoded_time])

        if self.individual:
            seasonal_output = tf.concat([tf.expand_dims(self.Linear_Seasonal[i](x[:,:,i]), axis=-1) for i in range(self.channels)], axis=-1)
            trend_output = tf.concat([tf.expand_dims(self.Linear_Trend[i](x[:,:,i]), axis=-1) for i in range(self.channels)], axis=-1)
            x = seasonal_output + trend_output
        else:
            seasonal_init, trend_init = tf.transpose(seasonal_init, perm=[0,2,1]), tf.transpose(trend_init, perm=[0,2,1])
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
            x = seasonal_output + trend_output
            x = tf.transpose(x, perm=[0,2,1]) # to [Batch, Output length, Channel]
            others_feature = self.Others_Dense(tf.transpose(features_combined, perm=[0,2,1]))
            others_feature = tf.transpose(others_feature, perm=[0,2,1])
            
            x = tf.concat([others_feature, x], axis=-1)
        # print(x.shape)
        if self.pred_len==1: x = tf.squeeze(self.final_layer(x), axis=-1)
        return x # [Batch, Output length, Channel]


class NLinear__Tensorflow(tf.keras.Model):
    """
    Normalization-Linear
    """
    def __init__(self, seq_len, pred_len, enc_in, individual):
        super(NLinear__Tensorflow, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
        self.individual = individual
        
        # Use this line if you want to visualize the weights
        # self.Linear.weights = (1/self.seq_len)*tf.ones([self.seq_len, self.pred_len])
        if self.individual:
            self.Linear = [Dense(self.pred_len) for _ in range(self.channels)]
        else:
            self.Linear = Dense(self.pred_len)
        self.final_layer = Dense(self.pred_len)

    def call(self, x):
        # x: [Batch, Input length, Channel]
        seq_last = x[:,-1:,:]
        x = x - seq_last
        if self.individual:
            x = tf.concat([tf.expand_dims(self.Linear[i](x[:,:,i]), axis=-1) for i in range(self.channels)], axis=-1)
        else:
            # print(x.shape)
            x = tf.transpose(x, perm=[0, 2, 1])
            x = self.Linear(x)
            x = tf.transpose(x, perm=[0, 2, 1])
        x = x + seq_last
        
        # print(tf.squeeze(self.final_layer(x), axis=-1).shape)
        if self.pred_len==1: x = tf.squeeze(self.final_layer(x), axis=-1)
        return x # [Batch, Output length, Channel]

    # def build(self, input_shape):
    #     super(NLinear__Tensorflow, self).build(input_shape)


class Embedded_NLinear__Tensorflow(tf.keras.Model):
    """
    Normalization-Linear
    """
    def __init__(self, seq_len, pred_len, enc_in, individual, seed):
        super(Embedded_NLinear__Tensorflow, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
        self.individual = individual
        
        # Use this line if you want to visualize the weights
        # self.Linear.weights = (1/self.seq_len)*tf.ones([self.seq_len, self.pred_len])
        if self.individual:
            self.Linear = [Dense(self.pred_len) for _ in range(self.channels)]
        else:
            self.Linear = Dense(self.pred_len)
        self.final_layer = Dense(self.pred_len)

        num_of_ID = 10_000
        input_len = seq_len[0]
        self.embedding_dim_size = 128
        # self.input_shape = (i, c)
        self.embedding_layer = layers.Embedding(input_dim=num_of_ID + 1, 
                                                    output_dim=self.embedding_dim_size, 
                                                    name='series_id_embedding_layer',
                                                    embeddings_initializer=initializers.RandomUniform(seed=seed))

        self.flat1 = layers.Flatten()
        self.repeat = layers.RepeatVector(input_len)

    def call(self, inputs):
        input_series_id = inputs[..., 0:1]
        input_encoded_time = inputs[..., 1:7]
        input_series = inputs[..., 7:]

        series_id_embedding = self.embedding_layer(input_series_id[:, 0:1,:])
        
        series_id_embedded = self.flat1(series_id_embedding)
        
        # x: [Batch, Input length, Channel]
        seq_last = input_series[:,-1:,:]
        sub = input_series - seq_last
        features_combined = layers.concatenate([self.repeat(series_id_embedded),  input_encoded_time, sub])
        if self.individual:
            x = tf.concat([tf.expand_dims(self.Linear[i](x[:,:,i]), axis=-1) for i in range(self.channels)], axis=-1)
        else:
            # print(x.shape)
            x = tf.transpose(features_combined, perm=[0, 2, 1])
            x = self.Linear(x)
            x = tf.transpose(x, perm=[0, 2, 1])
        # x = x + seq_last
        # add_ = x[..., :, 7:] + seq_last
        x = tf.concat([x[..., 0:128], x[..., 128:134], x[...,134:] + seq_last], axis=-1)
        # x = tf.concat([x[..., 0:self.embedding_dim_size], x[..., self.embedding_dim_size:134], x[...,134:] + seq_last], axis=-1)
        # print(tf.squeeze(self.final_layer(x), axis=-1).shape)
        if self.pred_len==1: x = tf.squeeze(self.final_layer(x), axis=-1)
        return x # [Batch, Output length, Channel]

    # def build(self, input_shape):
    #     super(NLinear__Tensorflow, self).build(input_shape)


class Linear__Tensorflow(tf.keras.Model):
    """
    Just one Linear layer
    """
    def __init__(self, seq_len, pred_len, enc_in, individual):
        super(Linear__Tensorflow, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
        self.individual = individual
        if self.individual:
            self.Linear = []
            for i in range(self.channels):
                self.Linear.append(Dense(units=self.pred_len))
        else:
            self.Linear = Dense(units=self.pred_len)
        self.final_layer = Dense(1)

    def call(self, x):
        # x: [Batch, Input length, Channel]
        if self.individual:
            x = tf.concat([tf.expand_dims(self.Linear[i](x[:,:,i]), axis=-1) for i in range(self.channels)], axis=-1)
            # output = tf.zeros(shape=[x.shape[0], self.pred_len, x.shape[2]], dtype=x.dtype)
            # for i in range(self.channels):
            #     output[:,:,i] = self.Linear[i](x[:,:,i])
            # x = output
        else:
            x = self.Linear(tf.transpose(x, perm=[0,2,1]))
            x = tf.transpose(x, perm=[0,2,1])
        if self.channels==1: x = tf.squeeze(self.final_layer(x), axis=-1)
        # print(f'{x.shape = }')
        return x # [Batch, Output length, Channel]
from keras import layers, models, initializers

class Embedded_Linear__Tensorflow(tf.keras.Model):
    """
    Just one Linear layer
    """
    def __init__(self, seq_len, pred_len, enc_in, individual, seed):
        super(Embedded_Linear__Tensorflow, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
        self.individual = individual
        if self.individual:
            self.Linear = []
            for i in range(self.channels):
                self.Linear.append(Dense(units=self.pred_len))
        else:
            self.Linear = Dense(units=self.pred_len)
        self.final_layer = Dense(1)

        num_of_ID = 10_000
        input_len = seq_len[0]
        embedding_dim_size = 128
        # self.input_shape = (i, c)
        self.embedding_layer = layers.Embedding(input_dim=num_of_ID + 1, 
                                                    output_dim=embedding_dim_size, 
                                                    name='series_id_embedding_layer',
                                                    embeddings_initializer=initializers.RandomUniform(seed=seed))

        self.flat1 = layers.Flatten()
        self.repeat = layers.RepeatVector(input_len)

    def call(self, inputs):
        # x: [Batch, Input length, Channel]
        
        input_series_id = inputs[..., 0:1]
        input_encoded_time = inputs[..., 1:7]
        input_series = inputs[..., 7:]

        series_id_embedding = self.embedding_layer(input_series_id[:, 0:1,:])
        series_id_embedded = self.flat1(series_id_embedding)
        features_combined = layers.concatenate([self.repeat(series_id_embedded), input_encoded_time, input_series])
        if self.individual:
            x = tf.concat([tf.expand_dims(self.Linear[i](x[:,:,i]), axis=-1) for i in range(self.channels)], axis=-1)
            # output = tf.zeros(shape=[x.shape[0], self.pred_len, x.shape[2]], dtype=x.dtype)
            # for i in range(self.channels):
            #     output[:,:,i] = self.Linear[i](x[:,:,i])
            # x = output
        else:
            x = self.Linear(tf.transpose(features_combined, perm=[0,2,1]))
            x = tf.transpose(x, perm=[0,2,1])
        if self.channels==1: x = tf.squeeze(self.final_layer(x), axis=-1)
        # print(f'{x.shape = }')
        return x # [Batch, Output length, Channel]

class LTSF_Linear__Tensorflow(LTSF_Linear_Base):
    def build(self):
        self.model = Linear__Tensorflow(seq_len=self.input_shape, 
                                        pred_len=self.output_shape, 
                                        enc_in=self.enc_in, 
                                        individual=self.individual)

class LTSF_Embedded_Linear__Tensorflow(LTSF_Linear_Base):
    def build(self):
        self.model = Embedded_Linear__Tensorflow(seq_len=self.input_shape, 
                                        pred_len=self.output_shape, 
                                        enc_in=self.enc_in, 
                                        individual=self.individual,
                                        seed=self.seed)

class LTSF_NLinear__Tensorflow(LTSF_Linear_Base):
    def build(self):
        self.model = NLinear__Tensorflow(seq_len=self.input_shape, 
                                         pred_len=self.output_shape, 
                                         enc_in=self.enc_in, 
                                         individual=self.individual)
        # _ = self.model(tf.random.normal(shape=list(self.input_shape)))
        # self.model.build(input_shape=self.input_shape)
        # self.model.summary()

class LTSF_Embedded_NLinear__Tensorflow(LTSF_Linear_Base):
    def build(self):
        self.model = Embedded_NLinear__Tensorflow(seq_len=self.input_shape, 
                                         pred_len=self.output_shape, 
                                         enc_in=self.enc_in, 
                                         individual=self.individual,
                                         seed=self.seed)

class LTSF_DLinear__Tensorflow(LTSF_Linear_Base):
    def build(self):
        self.model = DLinear__Tensorflow(seq_len=self.input_shape, 
                                         pred_len=self.output_shape, 
                                         enc_in=self.enc_in, 
                                         individual=self.individual)

class LTSF_Embedded_DLinear__Tensorflow(LTSF_Linear_Base):
    def build(self):
        self.model = Embedded_DLinear__Tensorflow(seq_len=self.input_shape, 
                                         pred_len=self.output_shape, 
                                         enc_in=self.enc_in, 
                                         individual=self.individual,
                                         seed=self.seed)