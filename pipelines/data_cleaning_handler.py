import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal
from scipy.stats import zscore

class DataCleaner:
    """
    A class for cleaning and preprocessing pandas DataFrames.

    Example:
    -------
    >>> cleaner = DataCleaner()
    """

    def __init__(self):
        self.report = {}

    def remove_null_value(
        self,
        *,
        data: pd.DataFrame,
        remove: bool = False,
        numeric_strategy: Literal['mean', 'median'] = 'mean',
        categorical_strategy: str = 'mode',
        visualize_null: bool = False,
        imputation_thres: float = 50.0
    ) -> pd.DataFrame:
        """
        Handle null values in the DataFrame by either removing or imputing them.

        Args:
        ----
        - data (pd.DataFrame): collects a DataFrame to perform operations on.
        - remove (bool): If True, removes rows with null values. If False, fills them using specified strategies.
        - numeric_strategy (Literal['mean', 'median']): Strategy to fill numeric columns.
        - categorical_strategy (str): Strategy to fill categorical columns ('mode' by default).
        - visualize_null (bool): If True, displays a bar chart of null values per column.
        - imputation_thres (float): set threshold for null value imputation, any column above the threshold will removed

        Returns:
        -------
        - pd.DataFrame: Cleaned DataFrame.
        """
        
        missing_value_per_column = ((data.isnull().sum()/len(data))*100).round(2)

        for col in data.columns.tolist():
            if data[col].isnull().sum() != 0:
                if round((data[col].isnull().sum()/len(data)) * 100) <= imputation_thres:
                    if remove:
                        data = data.dropna(subset=col)
                    else:
                        if pd.api.types.is_numeric_dtype(data[col]):
                            if numeric_strategy == 'mean':
                                data[col] = data[col].fillna(data[col].mean())
                                print(f"Filled numerical column '{col}' using {numeric_strategy}")
                            elif numeric_strategy == 'median':
                                data[col] = data[col].fillna(data[col].median())
                                print(f"Filled numerical column '{col}' using {numeric_strategy}")
                        else:
                            if categorical_strategy == 'mode':
                                data[col] = data[col].fillna(data[col].mode()[0])
                                print(f"Filled categorical column '{col}' using {categorical_strategy}")
                else:
                    data = data.drop(col, axis= 1)
        

        if visualize_null:
            fig, ax = plt.subplots(figsize=(12, 6))
            missing_value_per_column[missing_value_per_column > 0.00].plot(kind='bar', edgecolor='k', color = 'skyblue', ax=ax)
            ax.set_title("Missing Values per Column")
            ax.set_ylabel("Count")
            ax.set_xlabel("Column")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

        return data

    def check_duplicate(self, *,data: pd.DataFrame, remove_duplicate: bool = False) -> pd.DataFrame:
        """
        Checks and optionally removes duplicated rows in the DataFrame.

        Args:
        ----
        - remove_duplicate (bool): If True, removes duplicate rows.

        Returns:
        -------
        - pd.DataFrame: DataFrame after duplicate handling.
        """
        before = len(data)
        if data.duplicated().any():
            print(f"Found {data.duplicated().sum()} duplicate rows.")
            if remove_duplicate:
                data = data.drop_duplicates()
                print("Duplicates removed.")
        else:
            print("No duplicate rows found.")
        after = len(data)
        self.report['Number_of_duplicated_values'] = before - after
        return data

    def outlier_handler(
        self,
        *,
        data: pd.DataFrame,
        method: Literal['IQR', 'Z-score'] = 'IQR',
        z_thresh: float = 3.0,
        remove_outlier: bool = False
    ) -> pd.DataFrame:
        """
        Detects and optionally removes outliers using IQR or Z-score methods.

        Args:
        ----
        - method (Literal['IQR', 'Z-score']): Outlier detection method.
        - z_thresh (float): Z-score threshold, only applicable if method='Z-score'.
        - remove_outlier (bool): If True, removes rows with outliers.

        Returns:
        -------
        - pd.DataFrame: DataFrame with or without outliers depending on the `remove_outlier` flag.
        """
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        mask = pd.Series([False] * len(data))

        if method == 'IQR':
            for col in numeric_cols:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = (data[col] < lower_bound) | (data[col] > upper_bound)
                mask |= outliers
                print(f"{col}: {outliers.sum()} IQR outliers found.")
        elif method == 'Z-score':
            for col in numeric_cols:
                z_scores = zscore(data[col])
                outliers = np.abs(z_scores) > z_thresh
                mask |= outliers
                print(f"{col}: {outliers.sum()} Z-score outliers found.")

        if remove_outlier:
            data = data[~mask]
            print(f"Removed {mask.sum()} outliers using method '{method}'.")

        return data if remove_outlier else data.assign(outlier_flag=mask)

    
    def show_report(self):
        return self.report