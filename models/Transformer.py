from models._base_ import TensorflowModel
from tensorflow import keras
from tensorflow.keras import layers

class VanillaTransformer__Tensorflow(TensorflowModel):
    def __init__(self, modelConfigs, input_shape, output_shape, save_dir, normalize_layer=None, seed=941, **kwargs):
        super().__init__(modelConfigs=modelConfigs, input_shape=input_shape, output_shape=output_shape, normalize_layer=normalize_layer, seed=seed, save_dir=save_dir)
        self.head_size = self.modelConfigs['head_size']
        self.num_heads = self.modelConfigs['num_heads']
        self.ff_dim = self.modelConfigs['ff_dim']
        self.num_transformer_blocks = self.modelConfigs['num_transformer_blocks']

    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        # Normalization and Attention
        x = layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(x, x)
        x = layers.Dropout(dropout)(x)
        res = x + inputs

        # Feed Forward Part
        x = layers.LayerNormalization(epsilon=1e-6)(res)
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        return x + res

    def build(self):
        inputs = keras.Input(shape=self.input_shape)
        x = inputs
        for _ in range(self.num_transformer_blocks):
            x = self.transformer_encoder(x, self.head_size, self.num_heads, self.ff_dim, self.dropouts[0])

        x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
        for dim in self.units:
            x = layers.Dense(dim, activation="relu")(x)
            x = layers.Dropout(self.dropouts[1])(x)
        outputs = layers.Dense(self.output_shape, activation="softmax")(x)
        self.model =  keras.Model(inputs, outputs)
        self.model.summary()