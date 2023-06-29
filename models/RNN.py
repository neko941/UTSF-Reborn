import tensorflow as tf
from models._base_ import TensorflowModel
from keras import layers
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Bidirectional
from keras.initializers import GlorotUniform
from keras.initializers import RandomUniform

class VanillaRNN__Tensorflow(TensorflowModel):
    def body(self):
        # RNN Layer 1
        self.model.add(SimpleRNN(name='RNN_layer',
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

class EmbeddedRNN__Tensorflow(TensorflowModel):
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
                                                    embeddings_initializer=RandomUniform(seed=self.seed))(input_series_id[:,0:1,:])
        
        series_id_embedded = layers.Flatten()(series_id_embedding)
        features_combined = layers.concatenate([layers.RepeatVector(input_len)(series_id_embedded),  input_encoded_time, input_series])

        x = SimpleRNN(name='RNN_layer',
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

class BiRNN__Tensorflow(TensorflowModel):
    def body(self):
        # BiRNN Layer 1 
        self.model.add(Bidirectional(name='BiRNN_layer_1',
                                     layer=SimpleRNN(units=self.units[0], 
                                                     return_sequences=True,
                                                     kernel_initializer=GlorotUniform(seed=self.seed), 
                                                     activation=self.activations[0])))
        # BiRNN Layer 2 
        self.model.add(Bidirectional(name='BiRNN_layer_2',
                                     layer=SimpleRNN(units=self.units[1], 
                                                     return_sequences=True,
                                                     kernel_initializer=GlorotUniform(seed=self.seed), 
                                                     activation=self.activations[1])))
        # BiRNN Layer 3 
        self.model.add(Bidirectional(name='BiRNN_layer_3', 
                                     layer=SimpleRNN(units=self.units[2], 
                                                     return_sequences=False,
                                                     kernel_initializer=tf.initializers.GlorotUniform(seed=self.seed), 
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