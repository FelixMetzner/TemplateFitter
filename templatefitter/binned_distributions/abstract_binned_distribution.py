"""
Base class for binned distributions.
    Used as basis for
      - Templates (Fitting)
      - Histograms (Plotting)
    as these are basically the same thing, but require different
    methods and attributes for the specific two use-cases.
"""

import logging
import numpy as np

from typing import Union, Tuple, List
from scipy.stats import binned_statistic_dd

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "BinnedDistribution"
]


class BinnedDistribution:
    def __index__(self):
        self._name = None
        self._num_bins = None
        self._bin_edges = None
        self._bin_mids = None
        self._shape = None
        self._range = None
        self._bin_counts = None
        self._dimensions = None

    def fill(self, input_data, weights):
        # TODO
        # use binned_statistic_dd
        # for general case as done by Max specifically for 1d and 2d case.
        pass

    """
    def max_fill_1d(self, input_data, weights):
        if weights is None:
            weights = np.ones_like(input_data)
        if isinstance(weights, list):
            weights = np.array(weights)

        self._bin_counts += binned_statistic(
            input_data,
            weights,
            statistic='sum',
            bins=self._bin_edges,
            range=self._range
        )[0]

        self._bin_errors_sq += binned_statistic(
            input_data,
            weights ** 2,
            statistic='sum',
            bins=self._bin_edges,
            range=self._range
        )[0]

        self._is_empty = False

    def max_fill_2d(self, input_data, weights):
        if weights is None:
            weights = np.ones(len(input_data[0]))
        if isinstance(weights, list):
            weights = np.array(weights)

        self._bin_counts += binned_statistic_2d(
            x=input_data[0], y=input_data[1], values=weights, statistic="sum", bins=self._bin_edges
        )[0]

        self._bin_errors_sq += binned_statistic_2d(
            x=input_data[0],
            y=input_data[1],
            values=weights ** 2,
            statistic="sum",
            bins=self._bin_edges,
        )[0]
        self._is_empty = False
    """

    @classmethod
    def fill_from_binned(cls):
        # TODO
        pass

    # TODO: Maybe define classes for all of these basic properties...
    @property
    def name(self) -> Union[None, str]:
        """ Name of the distribution """
        return self._name

    @property
    def num_bins(self) -> Union[None, int]:
        """ Number of bins; flattened if multi-dimensional """
        return self._num_bins

    @property
    def bin_edges(self) -> Union[None, List[float], np.ndarray]:
        """ Bin edges; sum of (number of bins + 1) for each dimension """
        return self._bin_edges

    @property
    def bin_mids(self) -> Union[None, List[float], np.ndarray]:
        """ Central value for each bin """
        return self._bin_mids

    @property
    def shape(self) -> Union[None, int, Tuple[int, ...]]:
        """ Shape of the numpy array holding the binned distribution """
        return self._shape

    @property
    def range(self) -> Union[None, Tuple[float, float], Tuple[Tuple[float, float], ...]]:
        """ Lower and upper bound of each dimension of the binned distribution """
        return self._range

    @property
    def bin_counts(self) -> Union[None, np.ndarray]:
        """ The actual bin counts of the binned distribution """
        return self._bin_counts
