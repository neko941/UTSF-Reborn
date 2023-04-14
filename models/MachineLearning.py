from xgboost import XGBRegressor
from models._base_ import MachineLearningModel

class ExtremeGradientBoostingRegression(MachineLearningModel):
    def build(self):
        self.model = XGBRegressor(**self.modelConfigs)