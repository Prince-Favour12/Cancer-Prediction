from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

import pandas as pd
import numpy as np
from typing import Literal, Optional, Union, Tuple

DataType = Union[pd.DataFrame, np.ndarray]

class Transformation:
    """
    A class for performing data preprocessing tasks including scaling,
    encoding, and balancing for classification tasks.
    """

    def __init__(self):
        pass

    def scale(
        self,
        *,
        X_train: DataType,
        X_test: DataType,
        method: Literal['standard_scaler', 'min_max_scaler']
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if method == 'standard_scaler':
            scaler = StandardScaler()
        elif method == 'min_max_scaler':
            scaler = MinMaxScaler()
        else:
            raise ValueError("method must be 'standard_scaler' or 'min_max_scaler'")
        
        if isinstance(X_train, DataType):
            X_train = pd.DataFrame(X_train)
        if isinstance(X_test, DataType):
            X_test = pd.DataFrame(X_test)

        numeric_cols = X_train.select_dtypes(include=np.number).columns
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()

        X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

        return X_train_scaled, X_test_scaled

    def encode(
        self,
        *,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_encoded = X_train.copy()
        test_encoded = X_test.copy()

        cat_cols = train_encoded.select_dtypes(exclude=np.number).columns

        for col in cat_cols:
            if train_encoded[col].nunique() > 5:
                encoder = LabelEncoder()
                train_encoded[col] = encoder.fit_transform(train_encoded[col])
                test_encoded[col] = encoder.transform(test_encoded[col])
            else:
                train_encoded = pd.get_dummies(train_encoded, columns=[col], dtype=int)
                test_encoded = pd.get_dummies(test_encoded, columns=[col], dtype=int)
                train_encoded, test_encoded = train_encoded.align(test_encoded, join='left', axis=1, fill_value=0)

        return train_encoded, test_encoded

    def balance(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        method: Literal['smote', 'over', 'under'] = 'smote',
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Balances the dataset using the specified method:
        - 'smote': SMOTE oversampling
        - 'over': Random oversampling
        - 'under': Random undersampling

        Returns:
        - Tuple of resampled (X, y)
        """
        if method == 'smote':
            sampler = SMOTE(random_state=random_state)
        elif method == 'over':
            sampler = RandomOverSampler(random_state=random_state)
        elif method == 'under':
            sampler = RandomUnderSampler(random_state=random_state)
        else:
            raise ValueError("method must be one of 'smote', 'over', or 'under'")

        result = sampler.fit_resample(X, y)
        X_resampled, y_resampled = result[:2]
        return np.asarray(X_resampled), np.asarray(y_resampled)
