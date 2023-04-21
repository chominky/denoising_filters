import numpy as np
import pandas as pd
from typing import Union, List, Optional


def Guidedfilter(p: Union[np.ndarray, pd.Series],
                 I: Optional[Union[np.ndarray, pd.Series]] = None,
                 w: int = 31,
                 eps: int = 1000) -> Union[int, float]:
    """apply guided filter from given window seq

    Args:
        p (Union[np.ndarray, pd.Series]): the input time series
        I (Optional[Union[np.ndarray, pd.Series]]): the guidance time series
        w[int]: window size
        eps[int]: regularization parameter

    Returns:
        Union[int, float]: guided filtered value
    """
    if I is None:
        I = p
    if isinstance(p, pd.Series):
        p = np.array(p)
        I = np.array(I)
    weights = 1.0 / w * np.ones(w)
    mean_I = np.convolve(np.pad(I, (w//2, w//2), "edge"), weights, "valid")
    mean_p = np.convolve(np.pad(p, (w//2, w//2), "edge"), weights, "valid")
    cov_Ip = np.convolve(np.pad(I * p, (w//2, w//2), "edge"), weights, "valid") - mean_I * mean_p
    var_I = np.convolve(np.pad(I * I, (w//2, w//2), "edge"), weights, "valid") - mean_I * mean_I

    A = cov_Ip / (var_I + eps)
    b = mean_p - A * mean_I

    mean_A = np.convolve(np.pad(A, (w//2, w//2), "edge"), weights, "valid")
    mean_b = np.convolve(np.pad(b, (w//2, w//2), "edge"), weights, "valid")

    return mean_A * I + mean_b


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
        df[columns].apply(
        lambda x: filter_mapping[filter](x, w=window_size))
    
    return df

if __name__ == "__main__":
    df = pd.read_csv('./data/kospi.csv')
    df = apply_filter(df, columns=["close"], filter="guided", window_size=31)
