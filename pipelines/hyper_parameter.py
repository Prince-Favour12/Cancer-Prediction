from sklearn.model_selection import RandomSearchCV, GridSearchCV
from typing import Literal

class HyperParameter:
    def __init__(self, method: Literal['RandomSearch', 'GridSearch'] = None):
        self.method = method if (method != None) and (method in ['RandomSearch', 'GridSearch']) else raise ValueError(f"Invalid Type for method {method}")
        self.best_model = {}
        self.best_params = {}

    def hyper_tuning(self, X_train, y_train):
        pass

    def evaluation(self):
        pass
        
    def best_model(self):
        pass

    def best_params(self):
        pass

