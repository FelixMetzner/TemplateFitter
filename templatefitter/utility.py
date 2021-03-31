import os
import logging
import numpy as np

from itertools import islice
from numba import vectorize, float64, float32
from typing import Union, List, Iterable, AnyStr

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "PathType",
    "cov2corr",
    "corr2cov",
    "xlogyx",
    "get_systematic_cov_mat",
    "array_split_into",
]


PathType = Union[str, AnyStr, os.PathLike]


def cov2corr(cov: np.ndarray) -> np.ndarray:
    """
    Calculates the correlation matrix from a given
    covariance matrix.

    Arguments
    ---------
    cov : np.ndarray
        Covariance matrix. Shape is (n,n).

    Return
    ------
    out : np.ndarray
        Correlation matrix. Shape is (n,n).
    """
    d_inv = np.nan_to_num(np.diag(1.0 / np.sqrt(np.diag(cov))))
    return np.matmul(d_inv, np.matmul(cov, d_inv))


def corr2cov(corr: np.ndarray, var: np.ndarray) -> np.ndarray:
    """
    Calculates the covariance matrix from a given
    correlation matrix and a variance vector.

    Arguments
    ---------
    corr : np.ndarray
        Correlation matrix of shape (n,n).
    var : np.ndarray
        Variance vector of shape (n,).

    Return
    ------
    out : np.ndarray
        Covariance matrix. Shape is (n,n).
    """
    d_matrix = np.diag(var)
    return np.matmul(d_matrix, np.matmul(corr, d_matrix))


@vectorize([float32(float32, float32), float64(float64, float64)])
def xlogyx(x, y):
    """
    Compute :math:`x*log(y/x)`to a good precision when :math:`y~x`.
    The xlogyx function is taken from https://github.com/scikit-hep/probfit/blob/master/probfit/_libstat.pyx.
    """

    # TODO: Add a test of this xlogyx function to the unit tests!
    # Method 1
    if x < 1e-100 or y < 1e-100:
        return 0.0
    if x < y:
        res = x * np.log1p((y - x) / x)
    else:
        res = -x * np.log1p((x - y) / y)
    return res

    #  Method 2
    # result = np.where(x < y, x * np.log1p((y - x) / x), -x * np.log1p((x - y) / y))
    # return np.nan_to_num(result)

    #  Method 3
    # cond_list = [
    #     (x < y) & (x > 1e-100) & (y > 1e-100),
    #     (x > y) & (x > 1e-100) & (y > 1e-100)
    # ]
    # choice_list = [
    #     x * np.log1p((y - x) / x),
    #     -x * np.log1p((x - y) / y)
    # ]
    # result = np.select(condlist=cond_list, choicelist=choice_list, default=0.)
    # return result


def get_systematic_cov_mat(
        hup: np.ndarray,
        hdown: np.ndarray,
) -> np.ndarray:
    """
    Calculates covariance matrix from systematic variations
    for a histogram.

    Returns
    -------
    Covariance Matrix : numpy.ndarray
        Shape is (`num_bins`, `num_bins`).
    """
    diff_sym = (hup - hdown) / 2.0

    return np.outer(diff_sym, diff_sym)


def array_split_into(
        iterable: Iterable,
        sizes: List[int],
) -> np.ndarray:
    """
    Yields a list of arrays of size `n` from array iterable for each `n` in `sizes`.
    """

    itx = iter(iterable)

    for size in sizes:
        if size is None:
            yield np.array(list(itx))
            return
        else:
            yield np.array(list(islice(itx, size)))
