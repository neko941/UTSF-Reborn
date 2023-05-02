import os
import numpy as np
from sklearn.metrics import r2_score

def MAE(y_true, y_pred):
    """ Mean Absolute Error """
    return np.mean(np.abs((y_true - y_pred)))

def MSE(y_true, y_pred):
    """ Mean Squared Error """ 
    return np.mean((y_true - y_pred) ** 2)

def RMSE(y_true, y_pred):
    """ Root Mean Squared Error """
    return np.sqrt(np.mean((y_true-y_pred)**2))

def MAPE(y_true, y_pred):
    """ Mean Absolute Percentage Error """
    return np.mean(np.abs((y_true-y_pred) / (y_true + 1e-10))) * 100

def SMAPE(y_true, y_pred):
    """ Symmetric Mean Absolute Percentage Error """
    return np.mean(np.abs(y_pred - y_true) / ((np.abs(y_pred) + np.abs(y_true))/2)) * 100 

def R2(y_true, y_pred):
    # return 1 - (np.sum(np.power(y - yhat, 2)) / np.sum(np.power(y - np.mean(y), 2)))
    # 1 - np.sum(np.square(y_true - y_pred)) / np.sum(np.square(y_true - np.mean(y_true)))
    return r2_score(y_true, y_pred)

metric_dict = {
    'MAE' : MAE, 
    'MSE' : MSE,
    'RMSE' : RMSE, 
    'MAPE' : MAPE, 
    'SMAPE' : SMAPE,
    'R2' : R2
}

def score(y, yhat, r, path=None, model=''):
    if len(yhat.shape) == 3: 
        nsamples, nx, ny = yhat.shape
        yhat = yhat.reshape((nsamples,nx*ny))
    if r != -1:
        results = [str(np.round(np.float64(metric_dict[key](y, yhat)), r)) for key in metric_dict.keys()]
    else:
        results = [str(metric_dict[key](y, yhat)) for key in metric_dict.keys()]
    if path: 
        os.makedirs(os.path.join(path, 'values'), exist_ok=True)
        np.save(open(os.path.join(path, 'values', f'yhat-{model}.npy'), 'wb'), yhat)
    return results
