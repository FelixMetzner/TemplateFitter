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

from templatefitter.binned_distributions.binning import Binning

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["BinnedDistribution"]


class BinnedDistribution:
    def __init__(self, bins, dimensions, scope=None, name=None):
        self._name = name
        self._dimensions = dimensions
        self._binning = Binning(bins=bins, dimensions=dimensions, scope=scope)

        self._bin_counts = None
        self._bin_errors_sq = None
        self._shape = None

        self._is_empty = True

    def fill(self, input_data, weights, scope=None):
        prepared_weights = self.get_weights(in_weights=weights)

        # TODO: Take care that sample, values, bins and range are of the correct shape
        # TODO: Check the shapes!

        self._bin_counts += binned_statistic_dd(
            sample=input_data,
            values=prepared_weights,
            statistic='sum',
            bins=self._bin_edges,
            range=self._range
        )[0]

        self._bin_errors_sq += binned_statistic_dd(
            sample=input_data,
            values=prepared_weights ** 2,
            statistic='sum',
            bins=self._bin_edges,
            range=self._range
        )[0]

        self.is_empty = False

    @classmethod
    def fill_from_binned(cls, bin_counts, bin_edges, dimensions, bin_errors=None):
        instance = cls(bins=bin_edges, dimensions=dimensions)
        # TODO: Check stuff, e.g. dimensions and binning vs. shape of bin_counts...
        instance._bin_counts = bin_counts

        if bin_errors is None:
            bin_errors = np.sqrt(bin_counts)

        instance._bin_errors_sq = bin_errors ** 2
        instance.is_empty = False
        return instance

    # TODO: Maybe define classes for all of these basic properties...
    @property
    def name(self) -> Union[None, str]:
        """ Name of the distribution """
        return self._name

    @property
    def num_bins(self) -> int:
        """ Number of bins; flattened if multi-dimensional """
        return self._binning.num_bins_total()

    @property
    def bin_edges(self) -> np.ndarray:
        """ Bin edges; Length = sum of (number of bins + 1) for each dimension """
        return self._binning.bin_edges_flattened

    @property
    def bin_mids(self) -> Tuple[Tuple[float, ...]]:
        """ Central value for each bin """
        return self._binning.bin_mids

    @property
    def shape(self) -> Union[None, int, Tuple[int, ...]]:
        """ Shape of the numpy array holding the binned distribution """
        # TODO
        return self._shape

    @property
    def range(self) -> Tuple[Tuple[float, float], ...]:
        """ Lower and upper bound of each dimension of the binned distribution """
        return self._binning.range

    @property
    def bin_counts(self) -> Union[None, np.ndarray]:
        """ The actual bin counts of the binned distribution """
        return self._bin_counts

    @property
    def bin_errors_sq(self) -> Union[None, np.ndarray]:
        """ The squared errors on the bin counts of the binned distribution """
        return self._bin_errors_sq

    @property
    def is_empty(self) -> bool:
        """ Boolean indicating if the binned distribution is empty or filled """
        return self._is_empty

    @is_empty.setter
    def is_empty(self, value):
        assert self._is_empty is True, "Trying to reset is_empty flag."
        assert value is False, "Trying to reset is_empty flag."
        self._is_empty = value
