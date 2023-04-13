from models.LSTM import BiLSTM__Tensorflow
from models.Transformer import VanillaTransformer__Tensorflow

model_dict = [
{ 
    'model' : BiLSTM__Tensorflow,
    'help' : '',
    'type' : 'Tensorflow',
    'config' : r'.\configs\models\DeepLearning\BiLSTM__Tensorflow.yaml'
# },{
#     'model' : VanillaTransformer__Tensorflow,
#     'help' : '',
#     'type' : 'Tensorflow',
#     'config' : r'.\configs\models\DeepLearning\VanillaTransformer__Tensorflow.yaml'
}]