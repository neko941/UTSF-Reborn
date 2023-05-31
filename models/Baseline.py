from models._base_ import TensorflowModel
import tensorflow as tf
from tensorflow.keras import layers, models, initializers

class Baseline1__Tensorflow(TensorflowModel):
    def build(self):
        input_series_value = layers.Input(shape=(input_len, chanels), name='input_series_value')
        conv1d_diff = tf.keras.layers.Conv1D(filters=1, kernel_size=2, kernel_initializer=lambda dtype=None: tf.constant([[[1]], [[-1]]], dtype=dtype), bias_initializer='zeros')
        conv1d_diff.trainable = False 
        # input_series_value = tf.keras.layers.Input(shape=input_shape)
        diff = conv1d_diff(input_series_value)
        mean_local_sum = tf.reduce_mean(diff, axis=1)
        mean_samples = tf.reduce_mean(input_series_value, axis=1)
        last_sign = mean_samples - input_series_value[:,-1,:]
        outputs = tf.where(last_sign > 0,  input_series_value[:,-1,:] + mean_local_sum, input_series_value[:,-1,:] - mean_local_sum)

        return tf.keras.models.Model(inputs=[ input_series_value], outputs=[outputs])

