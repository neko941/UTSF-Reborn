from keras import backend as K
from keras.layers.core import Activation
from keras.utils import get_custom_objects
from keras.layers import Layer
from abc import abstractmethod

activation_dict = {
                    'xsinsquared': Activation(lambda x: x + (K.sin(x)) ** 2),
                    'xsin': Activation(lambda x: x + (K.sin(x))),
                    'snake_a.5': Activation(lambda x: SnakeActivation(x=x, a=0.5)),
                    'snake_a1': Activation(lambda x: SnakeActivation(x=x, a=1)),
                    'snake_a5': Activation(lambda x: SnakeActivation(x=x, a=5)),
                    'srs_a5_b3': Activation(lambda x: _SoftRootSign(x=x, alpha=5.0, beta=3.0)),
                  }

def get_custom_activations():
    get_custom_objects().update(activation_dict)  

def SnakeActivation(x, alpha: float = 0.5):
    return x - K.cos(2*alpha*x)/(2*alpha) + 1/(2*alpha)

def _SoftRootSign(x, alpha: float = 5.0, beta:float = 3.0):
    return x / (x / alpha + K.exp(-x / beta))

class TrainableActivationFunction(Layer):
    def __init__(self, **kwargs):
        super(TrainableActivationFunction, self).__init__(**kwargs)

    @abstractmethod
    def build(self, *inputs):
        raise NotImplementedError 

    @abstractmethod
    def call(self, *inputs):
        raise NotImplementedError 

    @abstractmethod
    def get_config(self, *inputs):
        raise NotImplementedError 

class SoftRootSign(Layer):
    def __init__(self, alpha: float = 5.0, beta:float = 3.0, trainable=True, **kwargs):
        super(SoftRootSign, self).__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        self.trainable = trainable

    def build(self, input_shape):
        self.alpha_factor = K.variable(self.alpha,
                                       dtype=K.floatx(),
                                       name='alpha_factor')
        self.beta_factor = K.variable(self.beta,
                                      dtype=K.floatx(),
                                      name='beta_factor')
        if self.trainable:
            self._trainable_weights.append(self.alpha_factor)
            self._trainable_weights.append(self.beta_factor)

        super(SoftRootSign, self).build(input_shape)

    def call(self, inputs, mask=None):
        return _SoftRootSign(x=inputs, alpha=self.alpha_factor, beta=self.beta_factor)

    def get_config(self):
        config = {
                    'alpha': self.get_weights()[0] if self.trainable else self.alpha,
                    'beta': self.get_weights()[1] if self.trainable else self.beta,
                    'trainable': self.trainable
                 }
        base_config = super(SoftRootSign, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))