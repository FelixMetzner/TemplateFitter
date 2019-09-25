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
import pandas as pd

from typing import Union, Tuple, NamedTuple
from scipy.stats import binned_statistic_dd

from templatefitter.binned_distributions.weights import Weights
from templatefitter.binned_distributions.systematics import SystematicsInfo
from templatefitter.binned_distributions.binning import Binning, BinsInputType, ScopeInputType

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["BinnedDistribution", "BaseDataContainer"]


class BaseDataContainer(NamedTuple):
    data: np.ndarray
    weights: np.ndarray
    systematics: SystematicsInfo


class BinnedDistribution:
    # TODO: Include some method to apply adaptive binning once the distribution is filled.
    # TODO: Maybe we need a distribution component as well...

    def __init__(self, bins: BinsInputType, dimensions: int, scope: ScopeInputType = None, name: Optional[str] = None):
        self._name = name
        self._dimensions = dimensions
        self._binning = Binning(bins=bins, dimensions=dimensions, scope=scope)

        self._bin_counts = np.zeros(self.num_bins)
        self._bin_errors_sq = np.zeros(self.num_bins)
        self._shape = self._bin_counts.shape
        self._check_shapes()

        self._base_data = None
        self._is_empty = True

    def _check_shapes(self):
        assert self.shape == self.num_bins, (self.shape, self.num_bins)
        assert sum(self.shape) == self.num_bins_total, (self.shape, self.num_bins_total)

    def fill(self, input_data, weights):
        # TODO: Fix initialization of weights
        prepared_weights = self._get_weights(in_weights=weights)

        # TODO: Take care that sample, values, bins and range are of the correct shape
        # TODO: Check the shapes!

        self._bin_counts += binned_statistic_dd(
            sample=input_data,
            values=prepared_weights,
            statistic='sum',
            bins=self.bin_edges,
            range=self.range
        )[0]

        self._bin_errors_sq += binned_statistic_dd(
            sample=input_data,
            values=prepared_weights ** 2,
            statistic='sum',
            bins=self.bin_edges,
            range=self.range
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

    # TODO: Use this function
    def _get_base_info(self, in_data, in_weights, in_systematics):
        if isinstance(in_data, pd.Series):
            data = in_data.values
        elif isinstance(in_data, pd.DataFrame):
            data = in_data[self._variable.df_label].values
        elif isinstance(in_data, np.ndarray):
            data = in_data
        else:
            raise RuntimeError(f"Got unexpected type for data: {type(in_data)}.\n"
                               f"Should be one of pd.DataFrame, pd.Series, np.ndarray.")

        weights = Weights(weight_input=in_weights, data=data, data_input=in_data).get_weights()
        assert len(data) == len(weights)

        systematics = SystematicsInfo(
            in_sys=in_systematics,
            data=data,
            in_data=in_data,
            weights=weights
        )

        return BaseDataContainer(data=data, weights=weights, systematics=systematics)

    @property
    def name(self) -> Union[None, str]:
        """ Name of the distribution """
        return self._name

    @property
    def num_bins(self) -> Tuple[int, ...]:
        """ Number of bins; multiple values if multi-dimensional """
        return self._binning.num_bins

    @property
    def num_bins_total(self) -> int:
        """ Number of bins after flattening, so the total number of bins """
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
    def shape(self) -> Tuple[int, ...]:
        """ Shape of the numpy array holding the binned distribution """
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

    # TODO: This method should be available to find range of distribution of data
    #       especially for multiple-component- or multi-dimensional distributions
    # def _find_range_from_components(self) -> Tuple[float, float]:
    #     min_vals = list()
    #     max_vals = list()
    #
    #     for component in itertools.chain(*self._mc_components.values()):
    #         min_vals.append(np.amin(component.data))
    #         max_vals.append(np.amax(component.data))
    #
    #     return np.amin(min_vals), np.amax(max_vals)


    # def _get_bin_edges(self) -> Tuple[np.ndarray, np.ndarray, float]:
    #     """
    #     Calculates the bin edges for the histogram.
    #     :return: Bin edges.
    #     """
    #     if self._variable.has_scope():
    #         scope = self._variable.scope
    #     else:
    #         scope = self._find_range_from_components()
    #
    #     low, high = scope[0], scope[1]
    #
    #     if self._variable.use_logspace:
    #         assert low > 0, \
    #             f"Cannot use log-space for variable {self._variable.x_label} since the minimum value is <= 0."
    #         bin_edges = np.logspace(np.log10(low), np.log10(high), self._num_bins + 1)
    #     else:
    #         bin_edges = np.linspace(low, high, self._num_bins + 1)
    #
    #     bin_mids = (bin_edges[1:] + bin_edges[:-1]) / 2
    #     bin_width = bin_edges[1] - bin_edges[0]
    #     return bin_edges, bin_mids, bin_width
