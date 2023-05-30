import tensorflow as tf
from tensorflow.keras import layers, models, initializers

def build_Baseline_idea1(input_len, chanels):
    
    
    input_series_value = layers.Input(shape=(input_len, chanels), name='input_series_value')

    

    def my_init(shape, dtype=None):
        return tf.constant([[[1]], [[-1]]], dtype=dtype)
    
    conv1d_diff = tf.keras.layers.Conv1D(filters=1, kernel_size=2, kernel_initializer=my_init, bias_initializer='zeros')
    conv1d_diff.trainable = False 
    # input_series_value = tf.keras.layers.Input(shape=input_shape)
    diff = conv1d_diff(input_series_value)
    mean_local_sum = tf.reduce_mean(diff, axis=1)
    mean_samples = tf.reduce_mean(input_series_value, axis=1)
    last_sign = mean_samples - input_series_value[:,-1,:]
    outputs = tf.where(last_sign > 0,  input_series_value[:,-1,:] + mean_local_sum, input_series_value[:,-1,:] - mean_local_sum)

    return tf.keras.models.Model(inputs=[ input_series_value], outputs=[outputs])


def build_Baseline_idea2(input_len, chanels):
    
    
    input_series_value = layers.Input(shape=(input_len, chanels), name='input_series_value')

    

    def my_init(shape, dtype=None):
        return tf.constant([[[1]], [[-1]]], dtype=dtype)
    
    conv1d_diff = tf.keras.layers.Conv1D(filters=1, kernel_size=2, kernel_initializer=my_init, bias_initializer='zeros')
    conv1d_diff.trainable = False 
    # input_series_value = tf.keras.layers.Input(shape=input_shape)
    diff = conv1d_diff(input_series_value)
    local_sum = tf.reduce_sum(diff, axis=1)
    mean_samples = tf.reduce_mean(input_series_value, axis=1)
    last_sign = mean_samples - input_series_value[:,-1,:]
    outputs = tf.where(last_sign > 0,  input_series_value[:,-1,:] + local_sum, input_series_value[:,-1,:] - local_sum)

    return tf.keras.models.Model(inputs=[ input_series_value], outputs=[outputs])


def build_Baseline_idea4(input_len, chanels):
    
    
    input_series_value = layers.Input(shape=(input_len, chanels), name='input_series_value')

    

    def my_init(shape, dtype=None):
        return tf.constant([[[1]], [[-1]]], dtype=dtype)
    
    conv1d_diff = tf.keras.layers.Conv1D(filters=1, kernel_size=2, kernel_initializer=my_init, bias_initializer='zeros')
    conv1d_diff.trainable = False 
    # input_series_value = tf.keras.layers.Input(shape=input_shape)
    diff = conv1d_diff(input_series_value)
    mean_local_sum = tf.reduce_mean(diff, axis=1)
    outputs = input_series_value[:,-1,:] + mean_local_sum
    return tf.keras.models.Model(inputs=[ input_series_value], outputs=[outputs])


def build_Baseline_ave(input_len, chanels):
    
    
    input_series_value = layers.Input(shape=(input_len, chanels), name='input_series_value')

    

    outputs = tf.reduce_mean(input_series_value, axis=1)

    model = models.Model(inputs=[ input_series_value], outputs=[outputs], name='build_Baseline_ave')

    return model


def build_Baseline_5ave(input_len, chanels):
    
    
    input_series_value = layers.Input(shape=(input_len, chanels), name='input_series_value')

    

    outputs = tf.reduce_mean(input_series_value[:,-5:,:], axis=1)

    model = models.Model(inputs=[ input_series_value], outputs=[outputs])

    return model



def build_Bi_LSTSM(seed, input_len, predict_len, chanels):


    input_series_value = layers.Input(shape=(input_len, chanels), name='input_series_value')


    x = layers.Bidirectional(name='BiLSTM_layer_1',
                            layer=layers.LSTM(units=256,
                                    return_sequences=True,
                                    kernel_initializer=initializers.GlorotUniform(seed=seed),
                                    activation='tanh'))(input_series_value)

    x = layers.Bidirectional(name='BiLSTM_layer_4',
                                layer=layers.LSTM(units=128,
                                    return_sequences=True,
                                    kernel_initializer=initializers.GlorotUniform(seed=seed),
                                    activation='tanh'))(x)
    x = layers.Bidirectional(name='BiLSTM_layer_5',
                            layer=layers.LSTM(units=64,
                                    return_sequences=False,
                                    kernel_initializer=initializers.GlorotUniform(seed=seed),
                                    activation='tanh'))(x)


    x = layers.Dense(name='Fully_Connected_layer_2',
                    units=32,
                    kernel_initializer=initializers.GlorotUniform(seed=seed),
                    activation='relu')(x)

    outputs = layers.Dense(name='Output_layer',
                    units=predict_len, 
                    kernel_initializer=initializers.GlorotUniform(seed=seed),
                    activation=None)(x)

    model = models.Model(inputs=[input_series_value], outputs=[outputs])
    return model