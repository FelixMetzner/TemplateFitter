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

from collections.abc import Sequence as CollectionsABCSequence
from typing import Union, Optional, Tuple, List, NamedTuple, Sequence

from templatefitter.utility import cov2corr
from templatefitter.binned_distributions.weights import Weights, WeightsInputType
from templatefitter.binned_distributions.systematics import SystematicsInfo, SystematicsInputType
from templatefitter.binned_distributions.binning import Binning, BinsInputType, ScopeInputType, BinEdgesType, \
    LogScaleInputType

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["BinnedDistribution", "BaseDataContainer", "DataColumnNamesInput", "DataInputType"]

DataInputType = Union[pd.Series, pd.DataFrame, np.ndarray, Sequence[pd.Series], Sequence[np.ndarray]]
DataColumnNamesInput = Union[None, str, List[str]]


class BaseDataContainer(NamedTuple):
    data: np.ndarray
    weights: np.ndarray
    systematics: SystematicsInfo


class BinnedDistribution:
    # TODO: Include some method to apply adaptive binning once the distribution is filled.
    # TODO: Maybe we need a distribution component as well... At least for plotting.

    def __init__(
            self,
            bins: BinsInputType,
            dimensions: int,
            scope: ScopeInputType = None,
            log_scale_mask: LogScaleInputType = False,
            name: Optional[str] = None,
            data: Optional[DataInputType] = None,
            weights: WeightsInputType = None,
            systematics: SystematicsInputType = None,
            data_column_names: DataColumnNamesInput = None
    ) -> None:
        self._name = name
        self._binning = Binning(bins=bins, dimensions=dimensions, scope=scope, log_scale=log_scale_mask)

        self._bin_counts = np.zeros(self.num_bins)
        self._bin_errors_sq = np.zeros(self.num_bins)
        self._shape = self._bin_counts.shape
        self._check_shapes()

        self._data_column_names = None
        self._init_data_column_names(data_column_names=data_column_names, data=data)

        self._base_data = None
        self._is_empty = True

        self._bin_covariance_matrix = None
        self._bin_correlation_matrix = None

        if data is not None:
            self._base_data = self._get_base_info(in_data=data, in_weights=weights, in_systematics=systematics)
            self.is_empty = False

    def fill(
            self,
            input_data: DataInputType,
            weights: WeightsInputType = None,
            systematics: SystematicsInputType = None
    ) -> None:
        self._base_data = self._get_base_info(in_data=input_data, in_weights=weights, in_systematics=systematics)

        bins = [np.array(list(edges)) for edges in self.bin_edges]

        self._bin_counts += np.histogramdd(
            sample=self._base_data.data,
            weights=self._base_data.weights,
            bins=bins,
            range=self.range
        )[0]

        self._bin_errors_sq += np.histogramdd(
            sample=self._base_data.data,
            weights=self._base_data.weights ** 2,
            bins=bins,
            range=self.range
        )[0]

        self.is_empty = False

    @classmethod
    def fill_from_binned(
            cls,
            bin_counts: np.ndarray,
            bin_edges: BinEdgesType,
            dimensions: int,
            bin_errors: Optional[np.ndarray] = None,
            name: Optional[str] = None
    ) -> "BinnedDistribution":
        if not len(bin_edges) == dimensions:
            raise ValueError(
                f"Bin edges represent a different number of dimensions than provided!\n"
                f"Number of dimensions extracted from bin edges: {len(bin_edges)}\n"
                f"Provided number of dimensions: {dimensions}"
            )
        if len(bin_counts.shape) == 1:
            if not dimensions == 1:
                raise ValueError(f"Shape of bin_counts {bin_counts.shape} does not match dimensions {dimensions}!")
        elif len(bin_counts.shape) == 2:
            if not dimensions == bin_counts.shape[1]:
                raise ValueError(f"Shape of bin_counts {bin_counts.shape} does not match dimensions {dimensions}!")
        else:
            raise ValueError(f"Unexpected shape of provided bin_counts!\n"
                             f"Should be (length of dataset, dimensions) but is {bin_counts.shape}")

        if bin_errors is None:
            bin_errors = np.sqrt(bin_counts)
        else:
            if not bin_errors.shape == bin_counts.shape:
                raise ValueError(f"Shapes of provided bin_counts {bin_counts.shape} "
                                 f"and bin_errors {bin_errors.shape} does not match!")

        instance = cls(bins=bin_edges, dimensions=dimensions, name=name)
        instance._bin_counts = bin_counts
        instance._bin_errors_sq = bin_errors ** 2

        instance.is_empty = False
        return instance

    def _get_base_info(
            self,
            in_data: DataInputType,
            in_weights: WeightsInputType,
            in_systematics: SystematicsInputType
    ) -> BaseDataContainer:
        data = self.get_data_input(in_data=in_data, data_column_names=self.data_column_names)

        if len(data.shape) == 1:
            assert self.dimensions == 1, self.dimensions
        elif len(data.shape) == 2:
            assert self.dimensions == data.shape[1], (self.dimensions, data.shape, data.shape[1])
        else:
            raise RuntimeError(f"The data related to the distribution is of unexpected shape:\n"
                               f"Shape of data: {data.shape}\nDimensions: {self.dimensions}")

        weights = Weights(weight_input=in_weights, data=data, data_input=in_data).get_weights()
        assert len(data) == len(weights)

        systematics = SystematicsInfo(
            in_sys=in_systematics,
            data=data,
            in_data=in_data,
            weights=weights
        )

        return BaseDataContainer(data=data, weights=weights, systematics=systematics)

    @staticmethod
    def get_data_input(
            in_data: DataInputType,
            data_column_names: DataColumnNamesInput = None
    ) -> np.ndarray:
        if isinstance(in_data, pd.Series):
            data = in_data.values
        elif isinstance(in_data, pd.DataFrame):
            if data_column_names is None:
                raise ValueError("If data is provided as pandas data frame, data_column_names must be provided too!")
            assert all(c in in_data.columns for c in data_column_names), (data_column_names, in_data.columns)
            data = in_data[data_column_names].values
        elif isinstance(in_data, np.ndarray):
            data = in_data
        elif isinstance(in_data, CollectionsABCSequence):
            first_type = type(in_data[0])
            assert all(isinstance(d_in, first_type) for d_in in in_data), [type(d) for d in in_data]
            if all(isinstance(d_in, pd.Series) for d_in in in_data):
                assert all(len(d_in.index) == len(in_data[0].index) for d_in in in_data), [len(d) for d in in_data]
                data = np.stack([d_in.values for d_in in in_data]).T
            elif all(isinstance(d_in, np.ndarray) for d_in in in_data):
                assert all(len(d_in.shape) == 1 for d_in in in_data), [d_in.shape for d_in in in_data]
                assert all(d_in.shape == in_data[0].shape for d_in in in_data), [d_in.shape for d_in in in_data]
                data = np.stack([d_in for d_in in in_data]).T
            else:
                raise ValueError(f"You provided a sequence of objects of type {first_type.__name__} for the "
                                 f"argument 'data' / 'input_data', but this parameter expects inputs of the following "
                                 f"types:\n\tpandas.Series\n\tpandas.DataFrame\n\tnumpy.ndarray\n\t"
                                 f"sequence of pandas.Series\n\tsequence of numpy.ndarray")
        else:
            raise RuntimeError(f"Got unexpected type for data: {type(in_data)}.\n"
                               f"Should be one of pd.DataFrame, pd.Series, np.ndarray.")

        assert isinstance(data, np.ndarray), type(data)
        return data

    @property
    def name(self) -> Union[None, str]:
        """ Name of the distribution """
        return self._name

    @property
    def binning(self) -> Binning:
        return self._binning

    @property
    def num_bins(self) -> Tuple[int, ...]:
        """ Number of bins; multiple values if multi-dimensional """
        return self._binning.num_bins

    @property
    def num_bins_total(self) -> int:
        """ Number of bins after flattening, so the total number of bins """
        return self._binning.num_bins_total

    @property
    def bin_edges(self) -> BinEdgesType:
        """ Bin edges; Tuple of length = self.dimensions and containing tuples with bin edges for each dimension """
        return self._binning.bin_edges

    @property
    def bin_edges_flattened(self) -> np.ndarray:
        """ Bin edges flattened to one dimension; Length = sum of (number of bins + 1) for each dimension """
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
    def dimensions(self) -> int:
        """ Dimensions of the distribution """
        return self._binning.dimensions

    @property
    def bin_counts(self) -> Union[None, np.ndarray]:
        """ The actual bin counts of the binned distribution """
        return self._bin_counts

    @property
    def bin_errors_sq(self) -> Union[None, np.ndarray]:
        """ The squared errors on the bin counts of the binned distribution """
        return self._bin_errors_sq

    @property
    def bin_errors(self) -> Union[None, np.ndarray]:
        if self._bin_errors_sq is None:
            return None
        return np.sqrt(self._bin_errors_sq)

    @property
    def systematics(self) -> SystematicsInfo:
        return self._base_data.systematics

    @property
    def bin_covariance_matrix(self) -> np.ndarray:
        if self._bin_covariance_matrix is not None:
            return self._bin_covariance_matrix

        assert not self.is_empty

        num_bins_total = self.num_bins_total
        if len(self.systematics) == 0:
            return np.eye(num_bins_total)

        cov = np.zeros((num_bins_total, num_bins_total))

        for sys_info in self.systematics:
            cov += sys_info.get_covariance_matrix(
                data=self._base_data.data,
                weights=self._base_data.weights,
                binning=self._binning
            )

        assert len(cov.shape) == 2, cov.shape
        assert cov.shape[0] == cov.shape[1], cov.shape
        assert cov.shape[0] == self.num_bins_total, (cov.shape[0], self.num_bins_total)
        assert np.allclose(cov, cov.T, rtol=1e-05, atol=1e-08), cov  # Checking if covariance matrix is symmetric.

        self._bin_covariance_matrix = cov
        return cov

    @property
    def bin_correlation_matrix(self) -> np.ndarray:
        if self._bin_correlation_matrix is not None:
            return self._bin_correlation_matrix

        self._bin_correlation_matrix = cov2corr(self.bin_covariance_matrix)
        return self._bin_correlation_matrix

    @property
    def is_empty(self) -> bool:
        """ Boolean indicating if the binned distribution is empty or filled """
        return self._is_empty

    @is_empty.setter
    def is_empty(self, value):
        assert self._is_empty is True, "Trying to reset is_empty flag."
        assert value is False, "Trying to reset is_empty flag."
        self._is_empty = value

    @property
    def data_column_names(self) -> Optional[List[str]]:
        return self._data_column_names

    @data_column_names.setter
    def data_column_names(self, column_names: DataColumnNamesInput) -> None:
        if self._data_column_names is not None:
            raise RuntimeError(f"You are trying to reset data_column_names\n"
                               f"\tfrom: {self._data_column_names}\n\tto:   {column_names}")
        self._init_data_column_names(data_column_names=column_names, data=None)

    @property
    def base_data(self) -> BaseDataContainer:
        return self._base_data

    def get_projection_on(self, dimension: int) -> Tuple[np.ndarray, Binning]:
        if dimension < 0 or dimension >= self.dimensions:
            raise ValueError(f"Parameter 'dimension' must be in [0, {self.dimensions - 1}] "
                             f"as the distribution has {self.dimensions} dimensions! You provided {dimension}.")
        other_dimensions = tuple(dim for dim in range(self.dimensions) if dim != dimension)
        projected_bin_count = self.bin_counts.sum(axis=other_dimensions)
        assert len(projected_bin_count.shape) == 1, projected_bin_count.shape

        reduced_binning = Binning(bins=self.bin_edges[dimension], dimensions=1, scope=self.range[dimension])

        assert len(projected_bin_count) == self.num_bins[dimension], \
            (len(projected_bin_count), self.num_bins[dimension])

        return projected_bin_count, reduced_binning

    def _check_shapes(self) -> None:
        assert self.shape == self.num_bins, (self.shape, self.num_bins)
        assert sum(self.shape) == self.num_bins_total, (self.shape, self.num_bins_total)

    def _init_data_column_names(self, data_column_names: DataColumnNamesInput, data: Optional[DataInputType]):
        if isinstance(data_column_names, str):
            assert self.dimensions == 1, (data_column_names, self.dimensions)
            if isinstance(data, pd.DataFrame):
                assert data_column_names in data.columns, (data_column_names, data.columns)
            self._data_column_names = [data_column_names]
        elif isinstance(data_column_names, list):
            assert self.dimensions == len(data_column_names), (data_column_names, self.dimensions)
            assert all(isinstance(col_name, str) for col_name in data_column_names)
            if isinstance(data, pd.DataFrame):
                assert all(c_name in data.columns for c_name in data_column_names), (data_column_names, data.columns)
            self._data_column_names = data_column_names
        else:
            if data_column_names is not None:
                raise ValueError("Received unexpected input for parameter 'data_column_names'.\n"
                                 "This parameter should be a list of column names of columns of the "
                                 "pandas.DataFrame that can be provided via the argument 'data'.")
            self._data_column_names = None
