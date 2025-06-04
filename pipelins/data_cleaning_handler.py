import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal
from scipy.stats import zscore

class DataCleaner:
    """
    Args
    ---
    - data: collects a DataFrame instance

    Example:
    ---
    >>> data = pd.read_csv('file.csv')
    >>> df = DataCleaner(data)
    """
    def __init__(self, data: pd.DataFrame):
        self.df = data.copy()
        self.report = {}

    def remove_null_value(self, *,remove:bool = True, numeric_strategy: str= Literal['mean', 'median'], categorical_strtegy: str = 'mode', visualize_null: bool = False) -> pd.DataFrame:
        """
        Args
        ---
        - **remove**: *If `True` removes null values from the data frame. if set to `False` doesn't remove any null value*
        - **numeric_strategy**: *if `mean`  fills numerical columns with the mean value, `median and mode` can also be specified*
        - **categorical_strategy:** *default `mode` fills categorical column with the most occurent value*
        - **visualize_null:** *default `False` doesn't return any visualization, when set to `True` returns a bar chart of number of missing values in each column*

        Example
        ---
        >>> data = pd.read_csv('file.csv')
        >>> df = DataCleaner(data)
        >>> df.remove_null_value(numeric_strategy= 'mean', visualize_null=True)

        """
        before = len(self.df)
        missing_value_per_column = self.df.isnull().sum()
        cols = self.df.columns.tolist()
        for col in cols:
            if remove:
                if self.df.isnull().any():
                    if self.df[col].isnull().any():
                        if pd.api.types.is_numeric_dtype(self.df[col]):
                            if numeric_strategy == 'mean':
                                self.df[col] = self.df[col].fillna(self.df[col].mean())
                                print(f"Filled numerical column '{col}' with {numeric_strategy}")
                            elif numeric_strategy == 'median':
                                self.df[col] = self.df[col].fillna(self.df[col].median())
                                print(f"Filled numerical column '{col}' with {numeric_strategy}")
                        elif pd.api.types.is_object_dtype(self.df[col]):
                            if categorical_strtegy:
                                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
                                print(f"Filled categorical column '{col}' with {categorical_strtegy}")
                    else:
                        print(f"column without null value '{col}'")
                else:
                    print(f"DataFrame doesn't contain missing values")
        
        after = len(self.df)

        if visualize_null:
            fig, ax = plt.subplots(figsize = (12, 8))
            ax.bar(x = missing_value_per_column.index, y = missing_value_per_column.values, width= 0.8, edgecolor= 'k')
            ax.set_xlabel("Columns")
            ax.set_ylabel("Number of missing values")

            for index, value in enumerate(missing_value_per_column.values):
                plt.text(x = int(index), y = value+1, s = str(value), ha= 'center')

            plt.tight_layout()

        self.report['Number_of_missing_values'] = before - after

        return self.df


    def check_duplicate(self, *,remove_duplicate = False) -> pd.DataFrame:
        """
        checks for duplicates

        Args
        ---
        - **remove_duplicate**: *default `False` which doesn't remove the duplicate but when set to `True` removes the duplicate*

        Example
        ---
        >>> data = pd.read_csv('file.csv')
        >>> df = DataCleaner(data)
        >>> df.check_duplicate(remove_duplicate= True)
        """
        before = len(self.df)
        if self.df.duplicated().any():
            if remove_duplicate:
                self.df = self.df.drop_duplicates()

        else:
            print("No duplicate Found")

        after = len(self.df)

        self.report['Number_of_duplicated_values'] = before - after

        return self.df

    def outlier_handler(self, *, method = Literal['IQR', 'Z-score'], z_thresh=3) -> pd.DataFrame:
        """
        Checks and remove outlier from the dataset


        Args
        ---
        **method:** *specifying method for finding outliers*

        **remove_outlier:** *if `True` remove outlier completely from dataset, if atdefault doesn't remove it just checks for it*

        Example
        ---
        >>> data = pd.read_csv('file.csv')
        >>> df = DataCleaner(data)
        >>> df.outlier_handler(remove_outlier= False)
        """
        before = len(self.df)
        df_outliers = self.df.copy()
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        if method == 'IQR':
            for col in numeric_cols:
                # ----- IQR Method -----
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_outliers[f'{col}_IQR_outlier'] = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)

        elif method == 'Z-score':
            for col in numeric_cols:
                # ----- Z-Score Method -----
                z_scores = zscore(self.df[col])
                df_outliers[f'{col}_Z_outlier'] = np.abs(z_scores) > z_thresh
        

        return df_outliers



