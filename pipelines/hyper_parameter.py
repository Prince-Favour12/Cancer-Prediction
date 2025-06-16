from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from typing import Literal, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd, numpy as np

class HyperParameter:
    """A class to perform hyperparameter tuning using either Random Search or Grid Search."""
    def __init__(self, method: Optional[Literal['RandomSearch', 'GridSearch']] = None):
        if (method is not None) and (method in ['RandomSearch', 'GridSearch']):
            self.method = method
        else:
            raise ValueError(f"Invalid Type for method {method}")
        self._best_model = {}
        self._best_params = {}

    def __generate_models(self) -> dict:
        """
        Generate a dictionary of models with their respective hyperparameters.

        Returns:
            A dictionary where keys are model names and values are model instances.
        """
        return {
            'LogisticRegression': LogisticRegression(),
            'RandomForestClassifier': RandomForestClassifier(),
            'GradientBoostingClassifier': GradientBoostingClassifier(),
            'SVC': SVC(),
            'KNeighborsClassifier': KNeighborsClassifier(),
            'DecisionTreeClassifier': DecisionTreeClassifier(),
            'GaussianNB': GaussianNB(),
            'XGBClassifier': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        }
    
    def __params_generator(self) -> dict:
        """
        Generate a dictionary of hyperparameters for each model.

        Returns:
            A dictionary where keys are model names and values are dictionaries of hyperparameters.
        """
        return {
            'LogisticRegression': {
                'C': np.logspace(-4, 4, 20),
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            },
            'RandomForestClassifier': {
                'n_estimators': [10, 50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            },
            'GradientBoostingClassifier': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'SVC': {
                'C': np.logspace(-4, 4, 20),
                'kernel': ['linear', 'rbf', 'poly']
            },
            'KNeighborsClassifier': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance']
            },
            'DecisionTreeClassifier': {
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            },
            'GaussianNB': {},
            'XGBClassifier': {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5]
            }
        }

    def hyper_tuning(self, X_train, y_train)-> None:
        """
        perform hyperparameter tuning using the specified method.

        Args:
            X_train: Training features.
            y_train: Training labels.

        Returns:
            None
        """
        if self.method == 'RandomSearch':
            models = self.__generate_models()
            params = self.__params_generator()
            for model_name, model in models.items():
                param_dist = params.get(model_name, {})
                if param_dist:  # Only tune if there are parameters to search
                    search = RandomizedSearchCV(
                        estimator=model,
                        param_distributions=param_dist,
                        n_iter=100,
                        scoring='accuracy',
                        cv=3,
                        verbose=1,
                        random_state=42,
                        n_jobs=-1
                    )
                    search.fit(X_train, y_train)
                    self._best_model[model_name] = search.best_estimator_
                    self._best_params[model_name] = search.best_params_
                else:
                    model.fit(X_train, y_train)
                    self._best_model[model_name] = model
                    self._best_params[model_name] = {}
        elif self.method == 'GridSearch':
            models = self.__generate_models()
            params = self.__params_generator()
            for model_name, model in models.items():
                param_grid = params.get(model_name, {})
                if param_grid:
                    search = GridSearchCV(
                        estimator=model,
                        param_grid=param_grid,
                        scoring='accuracy',
                        cv=3,
                        verbose=1,
                        n_jobs=-1
                    )
                    search.fit(X_train, y_train)
                    self._best_model[model_name] = search.best_estimator_
                    self._best_params[model_name] = search.best_params_
                else:
                    model.fit(X_train, y_train)
                    self._best_model[model_name] = model
                    self._best_params[model_name] = {}
        else:
            raise ValueError(f"Invalid method {self.method}. Use 'RandomSearch' or 'GridSearch'.")

    def evaluation(self, X_test, y_test):
        """
        Evaluate the best model on the test set.

        Args:
            X_test: Test features.
            y_test: Test labels.

        Returns:
            A DataFrame containing evaluation metrics for each model.
        """
        if not self._best_model:
            raise ValueError("No model has been trained yet. Please run hyper_tuning first.")
        
        results = {}
        for model_name, model in self._best_model.items():
            y_pred = model.predict(X_test)
            results[model_name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred, average='weighted'),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted')
            }
        return pd.DataFrame(results).T
        
    def best_model(self, model_name: Optional[str] = None):
        """
        Retrieve the best model or a specific model by name.
        
        Args:
            model_name: Optional; if provided, returns the specific model by name.

        Returns:
            The best model or the specified model by name.
        """
        return self._best_model.get(model_name,{}) if model_name else self._best_model

    def best_params(self, model_name: Optional[str] = None):
        """
        Retrieve the best hyperparameters or specific parameters by model name.

        Args:
            model_name: Optional; if provided, returns the specific parameters for the model by name.

        Returns:
            The best hyperparameters or the specified parameters for the model by name.
        """
        return self._best_params

