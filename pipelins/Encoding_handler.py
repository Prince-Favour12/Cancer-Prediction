from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
from typing import Literal

class Transformation:
    """
    A class for performing data transformation tasks such as scaling and encoding
    on a pandas DataFrame.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initializes the Transformation object.

        Parameters:
        - data (pd.DataFrame): The input DataFrame to be transformed.
        """
        self.df = data.copy()

    def scale(self, *, choice: Literal['standard_scaler', 'min_max_scaler']) -> pd.DataFrame:
        """
        Applies feature scaling to numeric columns in the DataFrame.

        Parameters:
        - choice (Literal): The scaling method to use. Must be either:
            - 'standard_scaler': Standardize features by removing the mean and scaling to unit variance.
            - 'min_max_scaler': Scale features to a given range, default is [0, 1].

        Returns:
        - pd.DataFrame: The transformed DataFrame with scaled numeric columns.
        """
        numeric_features = self.df.select_dtypes(include=np.number).columns.tolist()

        if choice == 'standard_scaler':
            scaler = StandardScaler()
            self.df[numeric_features] = scaler.fit_transform(self.df[numeric_features])

        elif choice == 'min_max_scaler':
            scaler = MinMaxScaler()
            self.df[numeric_features] = scaler.fit_transform(self.df[numeric_features])

        else:
            raise ValueError(f"Invalid choice '{choice}'. Must be 'standard_scaler' or 'min_max_scaler'.")

        return self.df

    def encode(self) -> pd.DataFrame:
        """
        Encodes categorical features in the DataFrame.
        - Features with more than 5 unique values are label encoded.
        - Features with 5 or fewer unique values are one-hot encoded.

        Returns:
        - pd.DataFrame: The transformed DataFrame with encoded categorical features.
        """
        categorical_features = self.df.select_dtypes(exclude=np.number).columns.tolist()

        for col in categorical_features:
            if self.df[col].nunique() > 5:
                encoder = LabelEncoder()
                self.df[col] = encoder.fit_transform(self.df[col])
            else:
                self.df = pd.get_dummies(self.df, columns=[col], dtype=int)

        return self.df
