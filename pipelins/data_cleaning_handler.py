import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal
from scipy.stats import zscore

class DataCleaner:
    """
    A class for cleaning and preprocessing pandas DataFrames.

    Args:
    ----
    - data (pd.DataFrame): The DataFrame to be cleaned.

    Example:
    -------
    >>> data = pd.read_csv('file.csv')
    >>> cleaner = DataCleaner(data)
    """

    def __init__(self, data: pd.DataFrame):
        self.df = data.copy()
        self.report = {}

    def remove_null_value(
        self,
        *,
        remove: bool = True,
        numeric_strategy: Literal['mean', 'median'] = 'mean',
        categorical_strategy: str = 'mode',
        visualize_null: bool = False
    ) -> pd.DataFrame:
        """
        Handle null values in the DataFrame by either removing or imputing them.

        Args:
        ----
        - remove (bool): If True, removes rows with null values. If False, fills them using specified strategies.
        - numeric_strategy (Literal['mean', 'median']): Strategy to fill numeric columns.
        - categorical_strategy (str): Strategy to fill categorical columns ('mode' by default).
        - visualize_null (bool): If True, displays a bar chart of null values per column.

        Returns:
        -------
        - pd.DataFrame: Cleaned DataFrame.
        """
        missing_value_per_column = self.df.isnull().sum()

        if self.df.isnull().any().any():
            for col in self.df.columns:
                if self.df[col].isnull().any():
                    if remove:
                        self.df = self.df.dropna(subset=[col])
                        print(f"Dropped rows with missing values in column: {col}")
                    else:
                        if pd.api.types.is_numeric_dtype(self.df[col]):
                            if numeric_strategy == 'mean':
                                self.df[col] = self.df[col].fillna(self.df[col].mean())
                            elif numeric_strategy == 'median':
                                self.df[col] = self.df[col].fillna(self.df[col].median())
                            print(f"Filled numerical column '{col}' using {numeric_strategy}")
                        else:
                            if categorical_strategy == 'mode':
                                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
                                print(f"Filled categorical column '{col}' using {categorical_strategy}")
        else:
            print("No missing values found.")

        if visualize_null:
            fig, ax = plt.subplots(figsize=(12, 6))
            missing_value_per_column[missing_value_per_column > 0].plot(kind='bar', edgecolor='k', ax=ax)
            ax.set_title("Missing Values per Column")
            ax.set_ylabel("Count")
            ax.set_xlabel("Column")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

        return self.df

    def check_duplicate(self, *, remove_duplicate: bool = False) -> pd.DataFrame:
        """
        Checks and optionally removes duplicate rows in the DataFrame.

        Args:
        ----
        - remove_duplicate (bool): If True, removes duplicate rows.

        Returns:
        -------
        - pd.DataFrame: DataFrame after duplicate handling.
        """
        before = len(self.df)
        if self.df.duplicated().any():
            print(f"Found {self.df.duplicated().sum()} duplicate rows.")
            if remove_duplicate:
                self.df = self.df.drop_duplicates()
                print("Duplicates removed.")
        else:
            print("No duplicate rows found.")
        after = len(self.df)
        self.report['Number_of_duplicated_values'] = before - after
        return self.df

    def outlier_handler(
        self,
        *,
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
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        mask = pd.Series([False] * len(self.df))

        if method == 'IQR':
            for col in numeric_cols:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
                mask |= outliers
                print(f"{col}: {outliers.sum()} IQR outliers found.")
        elif method == 'Z-score':
            for col in numeric_cols:
                z_scores = zscore(self.df[col])
                outliers = np.abs(z_scores) > z_thresh
                mask |= outliers
                print(f"{col}: {outliers.sum()} Z-score outliers found.")

        if remove_outlier:
            self.df = self.df[~mask]
            print(f"Removed {mask.sum()} outliers using method '{method}'.")

        return self.df if remove_outlier else self.df.assign(outlier_flag=mask)
