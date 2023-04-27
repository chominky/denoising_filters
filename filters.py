import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
from scipy.stats import norm
from typing import Union, List, Optional


def MovingAverageFilter(p: Union[np.ndarray, pd.Series],
                        w: int = 31) -> np.ndarray:
    """Apply moving average filter to given seq

    Args:
        p (Union[np.ndarray, pd.Series]): the input time series
        w (int, optional): window size. Defaults to 31.

    Returns:
        np.ndarray: moving average filtered seq
    """
    weights = 1.0 / w * np.ones(w)
    mean_p = np.convolve(np.pad(p, (w-1), "edge"), weights, "valid")

    return mean_p


def Guidedfilter(p: Union[np.ndarray, pd.Series],
                 I: Optional[Union[np.ndarray, pd.Series]] = None,
                 w: int = 31,
                 eps: int = 1000) -> np.ndarray:
    """Apply guided filter to given seq

    Args:
        p (Union[np.ndarray, pd.Series]): the input time series
        I (Optional[Union[np.ndarray, pd.Series]]): the guidance time series. Defaults to None.
        w[int]: window size. Defaults to 31.
        eps[int]: regularization parameter. Defaults to 1000.

    Returns:
        np.ndarray: guided filtered seq
    """
    assert w % 2 == 1, "window size should be odd for guided filter."
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


def BilateralFilter(p: Union[np.ndarray, pd.Series],
                    w: int = 31) -> np.ndarray:
    """Apply bilateral filter to given seq

    Args:
        p (Union[np.ndarray, pd.Series]): the input time series
        w (int, optional): window size. Defaults to 31.

    Returns:
        np.ndarray: bilateral filtered seq
    """
    RATIO = 4
    sigma_d = w / (RATIO * 2)
    p = np.pad(p, (w//2, w//2), "edge")
    def cal_norm_pdf(array, scale):
        return norm.pdf(array, loc=array[w//2], scale=scale)

    weights = cal_norm_pdf(np.arange(w), scale=sigma_d)
    weights /= weights.sum()

    sigma_i = (p.max() - p.min()) / 100.0
    sliding_p = sliding_window_view(p, w)
    pixel_w = np.apply_along_axis(cal_norm_pdf, 1, sliding_p, scale=sigma_i)
    weights = weights * pixel_w
    weights /= np.sum(weights, axis=1).reshape(-1, 1)

    return np.sum(sliding_p * weights, axis=1)



def apply_filter(df: pd.DataFrame, columns: List[str],
                 filter_method: Union[List[str], str], window_size: int) -> pd.DataFrame:
    """Apply filter to dataframe

    Args:
        df (pd.DataFrame): dataframe to apply filter
        columns (List[str]): columns to apply filter
        filter_method (Union[List[str], str]): type of filter
        window_size (int): window size of filter

    Returns:
        pd.DataFrame: dataframe with filtered column
    """
    filter_mapping = {"guided": Guidedfilter,
                      "bilateral": BilateralFilter,
                      "moving_average": MovingAverageFilter,}
    if isinstance(filter_method, str):
        filter_method = [filter_method]
    for filter_name in filter_method:
        df[[f"{filter_name}_{col}" for col in columns]] = \
            df[columns].apply(
            lambda x: filter_mapping[filter_name](x, w=window_size))
    
    return df

if __name__ == "__main__":
    df = pd.read_csv('./data/kospi.csv')
    df = apply_filter(df, columns=["close"], filter_method="guided", window_size=31)
