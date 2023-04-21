import numpy as np
import pandas as pd
from typing import Union, List, Optional


def Guidedfilter(p: Union[np.ndarray, pd.Series],
                 I: Optional[Union[np.ndarray, pd.Series]] = None,
                 eps: int = 1000) -> Union[int, float]:
    """apply guided filter from given window seq

    Args:
        p (Union[np.ndarray, pd.Series]): the input time series
        I (Optional[Union[np.ndarray, pd.Series]]): the guidance time series
        eps[int]: regularization parameter

    Returns:
        Union[int, float]: guided filtered value
    """
    if I is None:
        I = p
    if isinstance(p, pd.Series):
        p = np.array(p)
        I = np.array(I)
    mean_I = np.mean(I)
    mean_p = np.mean(p)
    cov_Ip = np.mean(I * p) - mean_I * mean_p
    var_I = np.mean(I * I) - mean_I * mean_I

    A = cov_Ip / (var_I + eps)
    b = mean_p - A * mean_I

    mean_A = np.mean(A)
    mean_b = np.mean(b)

    return mean_A * I[len(I) // 2] + mean_b


def apply_filter(df: pd.DataFrame, columns: List[str],
                 filter: str, window_size: int) -> pd.DataFrame:
    """Apply filter to dataframe

    Args:
        df (pd.DataFrame): dataframe to apply filter
        columns (List[str]): columns to apply filter
        filter (str): type of filter
        window_size (int): window size of filter

    Returns:
        pd.DataFrame: dataframe with filtered column
    """
    filter_mapping = {"guided": Guidedfilter}
    df[[f"{filter}_{col}" for col in columns]] = \
        df[columns].rolling(
        window_size, center=True, min_periods=1
        ).apply(filter_mapping[filter])
    
    return df

if __name__ == "__main__":
    df = pd.read_csv('./data/kospi.csv')
    df = apply_filter(df, columns=["close"], filter="guided", window_size=31)
