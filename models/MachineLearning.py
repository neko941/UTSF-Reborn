from models._base_ import MachineLearningModel

import time
import numpy as np
import matplotlib.pyplot as plt
from xgboost import plot_importance
from utils.general import convert_seconds

""" Linear Models """
from sklearn.linear_model import LinearRegression
class LinearRegressionWrapper(MachineLearningModel):
    def build(self):
        self.model = LinearRegression(**self.modelConfigs)

from sklearn.linear_model import Ridge 
class RidgeWrapper(MachineLearningModel):
    def build(self):
        self.model = Ridge(**self.modelConfigs)

from sklearn.linear_model import RidgeCV 
class RidgeCVWrapper(MachineLearningModel):
    def build(self):
        self.model = RidgeCV(**self.modelConfigs)

from sklearn.linear_model import SGDRegressor 
class SGDRegressorWrapper(MachineLearningModel):
    def build(self):
        self.model = SGDRegressor(**self.modelConfigs)

from sklearn.linear_model import ElasticNet 
class ElasticNetWrapper(MachineLearningModel):
    def build(self):
        self.model = ElasticNet(**self.modelConfigs)

from sklearn.linear_model import ElasticNetCV 
class ElasticNetCVWrapper(MachineLearningModel):
    def build(self):
        self.model = ElasticNetCV(**self.modelConfigs)

from sklearn.linear_model import Lars 
class LarsWrapper(MachineLearningModel):
    def build(self):
        self.model = Lars(**self.modelConfigs)

from sklearn.linear_model import LarsCV 
class LarsCVWrapper(MachineLearningModel):
    def build(self):
        self.model = LarsCV(**self.modelConfigs)

from sklearn.linear_model import Lasso 
class LassoWrapper(MachineLearningModel):
    def build(self):
        self.model = Lasso(**self.modelConfigs)

from sklearn.linear_model import LassoCV 
class LassoCVWrapper(MachineLearningModel):
    def build(self):
        self.model = LassoCV(**self.modelConfigs)

from sklearn.linear_model import LassoLars 
class LassoLarsWrapper(MachineLearningModel):
    def build(self):
        self.model = LassoLars(**self.modelConfigs)

from sklearn.linear_model import LassoLarsCV 
class LassoLarsCVWrapper(MachineLearningModel):
    def build(self):
        self.model = LassoLarsCV(**self.modelConfigs)

from sklearn.linear_model import LassoLarsIC 
class LassoLarsICWrapper(MachineLearningModel):
    def build(self):
        self.model = LassoLarsIC(**self.modelConfigs)

from sklearn.linear_model import OrthogonalMatchingPursuit 
class OrthogonalMatchingPursuitWrapper(MachineLearningModel):
    def build(self):
        self.model = OrthogonalMatchingPursuit(**self.modelConfigs)

from sklearn.linear_model import OrthogonalMatchingPursuitCV 
class OrthogonalMatchingPursuitCVWrapper(MachineLearningModel):
    def build(self):
        self.model = OrthogonalMatchingPursuitCV(**self.modelConfigs)

from sklearn.linear_model import ARDRegression 
class ARDRegressionWrapper(MachineLearningModel):
    def build(self):
        self.model = ARDRegression(**self.modelConfigs)

from sklearn.linear_model import BayesianRidge 
class BayesianRidgeWrapper(MachineLearningModel):
    def build(self):
        self.model = BayesianRidge(**self.modelConfigs)

from sklearn.linear_model import HuberRegressor 
class HuberRegressorWrapper(MachineLearningModel):
    def build(self):
        self.model = HuberRegressor(**self.modelConfigs)

from sklearn.linear_model import QuantileRegressor 
class QuantileRegressorWrapper(MachineLearningModel):
    def build(self):
        self.model = QuantileRegressor(**self.modelConfigs)

from sklearn.linear_model import RANSACRegressor 
class RANSACRegressorWrapper(MachineLearningModel):
    def build(self):
        self.model = RANSACRegressor(**self.modelConfigs)

from sklearn.linear_model import TheilSenRegressor 
class TheilSenRegressorWrapper(MachineLearningModel):
    def build(self):
        self.model = TheilSenRegressor(**self.modelConfigs)

from sklearn.linear_model import PoissonRegressor 
class PoissonRegressorWrapper(MachineLearningModel):
    def build(self):
        self.model = PoissonRegressor(**self.modelConfigs)

from sklearn.linear_model import TweedieRegressor 
class TweedieRegressorWrapper(MachineLearningModel):
    def build(self):
        self.model = TweedieRegressor(**self.modelConfigs)

from sklearn.linear_model import GammaRegressor 
class GammaRegressorWrapper(MachineLearningModel):
    def build(self):
        self.model = GammaRegressor(**self.modelConfigs)


""" Ensenble Modes """
import gc
from xgboost import XGBRegressor
from utils.npy_utils import NpyFileAppend
from rich.progress import track
from multiprocessing import Pool
class ExtremeGradientBoostingRegression(MachineLearningModel):
    def build(self):
        self.model = XGBRegressor(**self.modelConfigs)

    def preprocessing(self, x, classifier=False):
    #     # os.remove(temp_file)
        if classifier: res = [i.flatten().astype(int) for i in x]
        else: 
    #         # flatten_func = np.vectorize(lambda arr: arr.flatten())
    #         # res = flatten_func(x)

            temp_file = self.path_value / 'temp_flatten.npy'
            writer = NpyFileAppend(temp_file) 
            for i in track(x):
                writer.append(i.flatten())
            # num_processes = 8
            # def flatten_element(element, writer):
            #     return writera.append(element.flatten())
            # chunk_size = len(x) // num_processes
            # chunks = [x[i:i+chunk_size] for i in range(0, len(x), chunk_size)]
            # with Pool(processes=num_processes) as pool:
            #     pool.map(flatten_element, (chunks, writer))
            writer.close()
            res = np.load(temp_file, mmap_mode='r+')


    #         # temp_file = self.path_value / 'temp_flatten.npy'
    #         # writer = NpyFileAppend(temp_file) 
    #         # for i in track(x):
    #         #     writer.append(i.flatten())
    #         # writer.close()
    #         # res = np.load(temp_file, mmap_mode='r+')
    #     else: res = [i.flatten() for i in x]
        return res

    def fit(self, X_train, y_train, time_as_int=False, **kwargs):
        start = time.time()
        y_train = np.ravel(self.preprocessing(y_train, self.is_classifier), order='C')
        
        # y_train = self.preprocessing(y_train, self.is_classifier)

        # temp_file = self.path_value / 'temp.npy'
        # ytrain_writer = NpyFileAppend(temp_file) 
        # chunk_size = 1000
        # with self.ProgressBar() as progress:
        #     for i in progress.track(range(len(y_train) // chunk_size + 1), description='PreprocessData'):
        #         start_idx = i * chunk_size
        #         end_idx = (i + 1) * chunk_size
        #         chunk = y_train[start_idx:end_idx]
        #         chunk = np.ravel(chunk, order='C') 
        #         ytrain_writer.append(chunk)
        #         del chunk
        #         gc.collect()
        # del y_train
        # ytrain_writer.close()
        # gc.collect()
        # y_train = np.load(temp_file, mmap_mode='r+')
        # y_train = np.ravel(y_train, order='C') 

        x_train = self.preprocessing(X_train)
        self.model.fit(X=x_train, 
                       y=y_train)
        self.time_used = time.time() - start
        if not time_as_int:
            self.time_used = convert_seconds(self.time_used)
        plot_importance(self.model, max_num_features=None)
        plt.savefig(fname=self.path_plot / 'XGBoost_Feature_Importance.png')
        np.save(file=self.path_value / f'xtrain-{self.__class__.__name__}.npy',
                arr=x_train)
                
from sklearn.ensemble import AdaBoostRegressor 
class AdaBoostRegressorWrapper(MachineLearningModel):
    def build(self):
        self.model = AdaBoostRegressor(**self.modelConfigs)

from sklearn.ensemble import BaggingRegressor 
class BaggingRegressorWrapper(MachineLearningModel):
    def build(self):
        self.model = BaggingRegressor(**self.modelConfigs)

from sklearn.ensemble import ExtraTreesRegressor 
class ExtraTreesRegressorWrapper(MachineLearningModel):
    def build(self):
        self.model = ExtraTreesRegressor(**self.modelConfigs)

from sklearn.ensemble import GradientBoostingRegressor 
class GradientBoostingRegressorWrapper(MachineLearningModel):
    def build(self):
        self.model = GradientBoostingRegressor(**self.modelConfigs)

from sklearn.ensemble import IsolationForest 
class IsolationForestWrapper(MachineLearningModel):
    def build(self):
        self.model = IsolationForest(**self.modelConfigs)

from sklearn.ensemble import RandomForestRegressor 
class RandomForestRegressorWrapper(MachineLearningModel):
    def build(self):
        self.model = RandomForestRegressor(**self.modelConfigs)

from sklearn.ensemble import RandomTreesEmbedding 
class RandomTreesEmbeddingWrapper(MachineLearningModel):
    def build(self):
        self.model = RandomTreesEmbedding(**self.modelConfigs)

from sklearn.ensemble import StackingRegressor 
class StackingRegressorWrapper(MachineLearningModel):
    def build(self):
        self.modelConfigs['estimators'] = [('lr', RidgeCV())]
        self.model = StackingRegressor(**self.modelConfigs)

from sklearn.ensemble import VotingRegressor 
class VotingRegressorWrapper(MachineLearningModel):
    def build(self):
        self.modelConfigs['estimators'] = [('lr', RidgeCV())]
        self.model = VotingRegressor(**self.modelConfigs)
        

from sklearn.ensemble import HistGradientBoostingRegressor 
class HistGradientBoostingRegressorWrapper(MachineLearningModel):
    def build(self):
        self.model = HistGradientBoostingRegressor(**self.modelConfigs)

""" Tree Machine Learning Models """
from sklearn.tree import DecisionTreeRegressor 
class DecisionTreeRegressorWrapper(MachineLearningModel):
    def build(self):
        self.model = DecisionTreeRegressor(**self.modelConfigs)

from sklearn.tree import ExtraTreeRegressor 
class ExtraTreeRegressorWrapper(MachineLearningModel):
    def build(self):
        self.model = ExtraTreeRegressor(**self.modelConfigs)

""" Support Vector Machine """
from sklearn.svm import LinearSVR 
class LinearSVRWrapper(MachineLearningModel):
    def build(self):
        self.model = LinearSVR(**self.modelConfigs)

from sklearn.svm import NuSVR 
class NuSVRWrapper(MachineLearningModel):
    def build(self):
        self.model = NuSVR(**self.modelConfigs)

from sklearn.svm import SVR 
class SVRWrapper(MachineLearningModel):
    def build(self):
        self.model = SVR(**self.modelConfigs)








