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

from abc import ABC, abstractmethod
from typing import Union, Optional, Tuple, List, NamedTuple, Sequence

from templatefitter.utility import cov2corr
from templatefitter.binned_distributions.weights import Weights, WeightsInputType
from templatefitter.binned_distributions.systematics import SystematicsInfo, SystematicsInputType
from templatefitter.binned_distributions.binning import (
    Binning,
    BinsInputType,
    ScopeInputType,
    BinEdgesType,
    LogScaleInputType,
)

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "BinnedDistribution",
    "BinnedDistributionFromData",
    "BinnedDistributionFromHistogram",
    "BaseDataContainer",
    "DataColumnNamesInput",
    "DataInputType",
]

DataInputType = Union[pd.Series, pd.DataFrame, np.ndarray, Sequence[pd.Series], Sequence[np.ndarray]]
DataColumnNamesInput = Union[None, str, List[str]]


class BaseDataContainer(NamedTuple):
    data: np.ndarray
    weights: np.ndarray
    systematics: SystematicsInfo


class BinnedDistribution(ABC):
    # TODO: Include some method to apply adaptive binning once the distribution is filled.

    def __init__(
        self,
        bins: BinsInputType,
        dimensions: int,
        scope: ScopeInputType = None,
        log_scale_mask: LogScaleInputType = False,
        name: Optional[str] = None,
        data: Optional[DataInputType] = None,
        data_column_names: DataColumnNamesInput = None,
    ) -> None:
        self._name = name  # type: Optional[str]
        self._binning = Binning(
            bins=bins,
            dimensions=dimensions,
            scope=scope,
            log_scale=log_scale_mask,
        )

        self._bin_counts = None  # type: Optional[np.ndarray]
        self._bin_errors_sq = None  # type: Optional[np.ndarray]

        self._data_column_names = None  # type: Optional[List[str]]
        self._init_data_column_names(
            data_column_names=data_column_names,
            data=data,
        )

        self._base_data = None  # type: Optional[BaseDataContainer]
        self._is_empty = True  # type: bool

        self._bin_covariance_matrix = None  # type: Optional[np.ndarray]
        self._bin_correlation_matrix = None  # type: Optional[np.ndarray]

    def fill(
        self,
        input_data: DataInputType,
        weights: WeightsInputType = None,
        systematics: SystematicsInputType = None,
        bin_errors_squared: Optional[np.ndarray] = None,
    ) -> None:
        raise NotImplementedError(
            "This method is not implemented for the abstract base class BinnedDistribution, "
            "as it depends on the specific versions of the child classes."
        )

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
        return self.binning.num_bins

    @property
    def num_bins_squeezed(self) -> Tuple[int, ...]:
        return self.binning.num_bins_squeezed

    @property
    def num_bins_total(self) -> int:
        """ Number of bins after flattening, so the total number of bins """
        return self.binning.num_bins_total

    @property
    def bin_edges(self) -> BinEdgesType:
        """ Bin edges; Tuple of length = self.dimensions and containing tuples with bin edges for each dimension """
        return self.binning.bin_edges

    @property
    def bin_edges_flattened(self) -> np.ndarray:
        """ Bin edges flattened to one dimension; Length = sum of (number of bins + 1) for each dimension """
        return self.binning.bin_edges_flattened

    @property
    def bin_mids(self) -> Tuple[Tuple[float, ...], ...]:
        """ Central value for each bin """
        return self.binning.bin_mids

    @property
    def shape(self) -> Tuple[int, ...]:
        """ Shape of the numpy array holding the binned distribution """
        return self.num_bins_squeezed

    @property
    def range(self) -> Tuple[Tuple[float, float], ...]:
        """ Lower and upper bound of each dimension of the binned distribution """
        return self.binning.range

    @property
    def dimensions(self) -> int:
        """ Dimensions of the distribution """
        return self.binning.dimensions

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
            raise RuntimeError(
                f"You are trying to reset data_column_names\n"
                f"\tfrom: {self._data_column_names}\n\tto:   {column_names}"
            )
        self._init_data_column_names(
            data_column_names=column_names,
            data=None,
        )

    @property
    def bin_counts(self) -> Optional[np.ndarray]:
        """ The actual bin counts of the binned distribution """
        return self._bin_counts

    @property
    def bin_errors_sq(self) -> Optional[np.ndarray]:
        """ The squared errors on the bin counts of the binned distribution """
        return self._bin_errors_sq

    @property
    def bin_errors(self) -> Optional[np.ndarray]:
        if self._bin_errors_sq is None:
            return None
        return np.sqrt(self._bin_errors_sq)

    @property
    @abstractmethod
    def systematics(self) -> SystematicsInfo:
        raise NotImplementedError(
            "This method is not implemented for the abstract base class BinnedDistribution, "
            "as it depends on the specific versions of the child classes."
        )

    @property
    @abstractmethod
    def bin_covariance_matrix(self) -> np.ndarray:
        raise NotImplementedError(
            "This method is not implemented for the abstract base class BinnedDistribution, "
            "as it depends on the specific versions of the child classes."
        )

    @property
    @abstractmethod
    def bin_correlation_matrix(self) -> np.ndarray:
        raise NotImplementedError(
            "This method is not implemented for the abstract base class BinnedDistribution, "
            "as it depends on the specific versions of the child classes."
        )

    @property
    @abstractmethod
    def base_data(self) -> BaseDataContainer:
        raise NotImplementedError(
            "This method is not implemented for the abstract base class BinnedDistribution, "
            "as it depends on the specific versions of the child classes."
        )

    @abstractmethod
    def bin_errors_sq_with_normalization(
        self,
        normalization_factor: Optional[float] = None,
    ) -> Union[None, np.ndarray]:
        raise NotImplementedError(
            "This method is not implemented for the abstract base class BinnedDistribution, "
            "as it depends on the specific versions of the child classes."
        )

    def get_projection_on(
        self,
        dimension: int,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Binning]:
        # TODO: Requires better treatment of the bin errors!
        projected_bin_count, projected_errors_sq = self.project_onto_dimension(
            bin_counts=self.bin_counts,
            dimension=dimension,
            bin_errors_squared=self.bin_errors_sq,
        )
        reduced_binning = self.binning.get_binning_for_one_dimension(dimension=dimension)

        return projected_bin_count, projected_errors_sq, reduced_binning

    def project_onto_dimension(
        self,
        bin_counts: np.ndarray,
        dimension: int,
        bin_errors_squared: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        # Covering 1-dimensional case where no projection is necessary.
        if self.dimensions == 1 and dimension == 0:
            return bin_counts, bin_errors_squared

        # TODO: The following part requires better treatment of the bin errors!
        if dimension < 0 or dimension >= self.dimensions:
            raise ValueError(
                f"Parameter 'dimension' must be in [0, {self.dimensions - 1}] "
                f"as the distribution has {self.dimensions} dimensions! You provided {dimension}."
            )
        other_dimensions = tuple(dim for dim in range(self.dimensions) if dim != dimension)
        projected_bin_count = bin_counts.sum(axis=other_dimensions)  # type: np.ndarray

        assert len(projected_bin_count.shape) == 1, projected_bin_count.shape
        assert len(projected_bin_count) == self.num_bins[dimension], (len(projected_bin_count), self.num_bins[dimension])

        if bin_errors_squared is not None:
            projected_errors_sq = bin_errors_squared.sum(axis=other_dimensions)
            assert len(projected_errors_sq.shape) == 1, projected_errors_sq.shape
            assert len(projected_errors_sq) == self.num_bins[dimension], (
                len(projected_errors_sq),
                self.num_bins[dimension],
            )
        else:
            projected_errors_sq = None

        return projected_bin_count, projected_errors_sq

    def project_onto_two_dimensions(
        self,
        bin_counts: np.ndarray,
        dimensions: Tuple[int, int],
        bin_errors_squared: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        # Covering 2-dimensional case where no projection is necessary.
        if self.dimensions == 2 and set(dimensions) == {0, 1}:
            return bin_counts, bin_errors_squared

        # TODO: The following part requires better treatment of the bin errors!
        if not (dimensions[0] < dimensions[1] and len(dimensions) == 2):
            raise ValueError(
                f"Parameter 'dimensions' must be tuple of two integers from [0, {self.dimensions - 1}] "
                f"as the distribution has {self.dimensions} dimensions!\n"
                f"Furthermore, the first integer must be smaller than the second!\n"
                f"You provided {dimensions}."
            )

        if not all(0 <= dim < self.dimensions for dim in dimensions):
            raise ValueError(
                f"Parameter 'dimensions' must be tuple of two integers from [0, {self.dimensions - 1}] "
                f"as the distribution has {self.dimensions} dimensions! You provided {dimensions}."
            )

        other_dimensions = tuple(dim for dim in range(self.dimensions) if dim not in dimensions)
        projected_bin_count = bin_counts.sum(axis=other_dimensions)  # type: np.ndarray

        assert len(projected_bin_count.shape) == 2, projected_bin_count.shape
        assert projected_bin_count.shape[0] == self.num_bins[dimensions[0]], (
            projected_bin_count.shape,
            self.num_bins[dimensions[0]],
        )
        assert projected_bin_count.shape[1] == self.num_bins[dimensions[1]], (
            projected_bin_count.shape,
            self.num_bins[dimensions[1]],
        )

        if bin_errors_squared is not None:
            projected_errors_sq = bin_errors_squared.sum(axis=other_dimensions)
            assert len(projected_errors_sq.shape) == 2, projected_bin_count.shape
            assert projected_errors_sq.shape[0] == self.num_bins[dimensions[0]], (
                projected_errors_sq.shape,
                self.num_bins[dimensions[0]],
            )
            assert projected_errors_sq.shape[1] == self.num_bins[dimensions[1]], (
                projected_errors_sq.shape,
                self.num_bins[dimensions[1]],
            )
        else:
            projected_errors_sq = None

        return projected_bin_count, projected_errors_sq

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
                raise ValueError(
                    "Received unexpected input for parameter 'data_column_names'.\n"
                    "This parameter should be a list of column names of columns of the "
                    "pandas.DataFrame that can be provided via the argument 'data'."
                )
            self._data_column_names = None


class BinnedDistributionFromData(BinnedDistribution):
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
        data_column_names: DataColumnNamesInput = None,
    ) -> None:
        super().__init__(
            bins=bins,
            dimensions=dimensions,
            scope=scope,
            log_scale_mask=log_scale_mask,
            name=name,
            data_column_names=data_column_names,
        )
        if data is not None:
            self.fill(
                input_data=data,
                weights=weights,
                systematics=systematics,
            )

    def fill(
        self,
        input_data: DataInputType,
        weights: WeightsInputType = None,
        systematics: SystematicsInputType = None,
        bin_errors_squared: Optional[np.ndarray] = None,
    ) -> None:
        assert self.is_empty
        if bin_errors_squared is not None:
            raise TypeError(
                "The parameter 'bin_errors_squared' is not to used for the initialization of the "
                "BinnedDistributionFromData class!"
            )

        self._base_data = self._get_base_info(
            in_data=input_data,
            in_weights=weights,
            in_systematics=systematics,
        )

        bins = [np.array(list(edges)) for edges in self.bin_edges]

        self._bin_counts = np.histogramdd(
            sample=self._base_data.data,
            weights=self._base_data.weights,
            bins=bins,
            range=self.range,
        )[0]
        assert self._bin_counts.shape == self.num_bins, (self._bin_counts.shape, self.num_bins)

        self._bin_errors_sq = np.histogramdd(
            sample=self._base_data.data,
            weights=np.square(self._base_data.weights),
            bins=bins,
            range=self.range,
        )[0]
        assert self._bin_errors_sq.shape == self.num_bins, (self._bin_errors_sq.shape, self.num_bins)

        self.is_empty = False

    def _get_base_info(
        self,
        in_data: DataInputType,
        in_weights: WeightsInputType,
        in_systematics: SystematicsInputType,
    ) -> BaseDataContainer:
        data = self.get_data_input(in_data=in_data, data_column_names=self.data_column_names)

        if len(data.shape) == 1:
            assert self.dimensions == 1, self.dimensions
        elif len(data.shape) == 2:
            assert self.dimensions == data.shape[1], (self.dimensions, data.shape, data.shape[1])
        else:
            raise RuntimeError(
                f"The data related to the distribution is of unexpected shape:\n"
                f"Shape of data: {data.shape}\nDimensions: {self.dimensions}"
            )

        weights = Weights(weight_input=in_weights, data=data, data_input=in_data).get_weights()
        assert len(data) == len(weights)

        systematics = SystematicsInfo(
            in_sys=in_systematics,
            data=data,
            in_data=in_data,
            weights=weights,
        )

        return BaseDataContainer(
            data=data,
            weights=weights,
            systematics=systematics,
        )

    @staticmethod
    def get_data_input(
        in_data: DataInputType,
        data_column_names: DataColumnNamesInput = None,
    ) -> np.ndarray:
        if isinstance(in_data, pd.Series):
            if data_column_names is not None and in_data.name not in data_column_names:
                logging.warning(
                    f"The data series name '{in_data.name}' is not in the data_column_names "
                    f"{data_column_names}.\nMake sure you provided the data for the correct variable."
                )
            data = in_data.values
        elif isinstance(in_data, pd.DataFrame):
            if data_column_names is None:
                raise ValueError("If data is provided as pandas data frame, data_column_names must be provided too!")
            if isinstance(data_column_names, str):
                assert data_column_names in in_data.columns, (data_column_names, in_data.columns)
            else:
                assert all(c in in_data.columns for c in data_column_names), (data_column_names, in_data.columns)
            data = in_data[data_column_names].values
        elif isinstance(in_data, np.ndarray):
            data = in_data
        elif isinstance(in_data, Sequence):
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
                raise ValueError(
                    f"You provided a sequence of objects of type {first_type.__name__} for the "
                    f"argument 'data' / 'input_data', but this parameter expects inputs of the following "
                    f"types:\n\tpandas.Series\n\tpandas.DataFrame\n\tnumpy.ndarray\n\t"
                    f"sequence of pandas.Series\n\tsequence of numpy.ndarray"
                )
        else:
            raise RuntimeError(
                f"Got unexpected type for data: {type(in_data)}.\n"
                f"Should be one of pd.DataFrame, pd.Series, np.ndarray."
            )

        assert isinstance(data, np.ndarray), type(data)
        return data

    @property
    def base_data(self) -> BaseDataContainer:
        assert self._base_data is not None
        return self._base_data

    @property
    def systematics(self) -> SystematicsInfo:
        return self.base_data.systematics

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
                data=self.base_data.data,
                weights=self.base_data.weights,
                binning=self.binning,
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

    def bin_errors_sq_with_normalization(self, normalization_factor: Optional[float] = None) -> Union[None, np.ndarray]:
        if self.is_empty:
            return None

        if normalization_factor is None or normalization_factor == 1.0:
            return self.bin_errors_sq

        bin_errors_sq = np.histogramdd(
            sample=self.base_data.data,
            weights=np.square(self.base_data.weights * normalization_factor),
            bins=[np.array(list(edges)) for edges in self.bin_edges],
            range=self.range,
        )[0]
        assert bin_errors_sq.shape == self.num_bins, (bin_errors_sq.shape, self.num_bins)
        return bin_errors_sq


class BinnedDistributionFromHistogram(BinnedDistribution):
    def __init__(
        self,
        bins: BinsInputType,
        dimensions: int,
        scope: ScopeInputType = None,
        log_scale_mask: LogScaleInputType = False,
        name: Optional[str] = None,
        data: Optional[DataInputType] = None,
        bin_errors_squared: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(
            bins=bins,
            dimensions=dimensions,
            scope=scope,
            log_scale_mask=log_scale_mask,
            name=name,
        )

        if data is not None:
            self.fill(
                input_data=data,
                bin_errors_squared=bin_errors_squared,
            )

    def fill(
        self,
        input_data: DataInputType,
        weights: WeightsInputType = None,
        systematics: SystematicsInputType = None,
        bin_errors_squared: Optional[np.ndarray] = None,
    ) -> None:
        assert self.is_empty

        info_text = (
            "The bin_count should be provided via the argument 'input_data' in form of a numpy ndarray.\n"
            "Its shape must fit the binning of the BinndedDistribution instance.\n"
            "The bin errors squared can be provided via 'bin_errors_squared' as np.ndarray and must also "
            "have the respective shape!"
        )
        if weights is not None:
            raise TypeError(
                f"The parameter 'weights' is not to used for the initialization of the "
                f"BinnedDistributionFromHistogram class!\n{info_text}"
            )
        if systematics is not None:
            raise TypeError(
                f"The parameter 'systematics' is not to used for the initialization of the "
                f"BinnedDistributionFromHistogram class!\n{info_text}"
            )

        bin_counts, bin_errors_sq = self.get_data_input(in_data=input_data, bin_errors_squared=bin_errors_squared)
        self._bin_counts = bin_counts
        self._bin_errors_sq = bin_errors_sq

        self.is_empty = False

    def get_data_input(
        self,
        in_data: DataInputType,
        bin_errors_squared: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:

        bin_counts = self.check_and_get_binned_input(binned_input=in_data, parameter_name="input_data")

        while bin_counts.shape[-1] == 1:
            axis = len(bin_counts.shape) - 1
            bin_counts = np.squeeze(bin_counts, axis=axis)

        assert bin_counts.shape == self.num_bins, (bin_counts.shape, self.num_bins)

        if bin_errors_squared is None:
            bin_errors_sq = bin_counts
        else:
            bin_errors_sq = self.check_and_get_binned_input(
                binned_input=bin_errors_squared,
                parameter_name="bin_errors_squared",
            )

        while bin_errors_sq.shape[-1] == 1:
            axis = len(bin_errors_sq.shape) - 1
            bin_errors_sq = np.squeeze(bin_errors_sq, axis=axis)

        assert bin_errors_sq.shape == self.num_bins, (bin_errors_sq.shape, self.num_bins)

        return bin_counts, bin_errors_sq

    def check_and_get_binned_input(
        self,
        binned_input: np.ndarray,
        parameter_name: str,
    ) -> np.ndarray:
        if not isinstance(binned_input, np.ndarray):
            raise TypeError(
                f"The argument '{parameter_name}' must be already histogrammed data of type numpy.ndarray "
                f"for BinnedDistributionFromHistogram!\n"
                f"However, you provided an object of type{type(binned_input)} instead!"
            )

        if len(binned_input.shape) == 1:
            if not self.dimensions == 1:
                raise ValueError(
                    f"Shape of {parameter_name} {binned_input.shape} does not match the "
                    f"dimensions of the BinndedDistribution instance, which is {self.dimensions}!"
                )
            return np.expand_dims(binned_input, axis=1)
        elif len(binned_input.shape) >= 2:
            if not self.dimensions == len(binned_input.shape):
                raise ValueError(
                    f"Shape of {parameter_name} {binned_input.shape} does not match the "
                    f"dimensions of the BinndedDistribution instance, which is {self.dimensions}!"
                )
            return binned_input
        else:
            raise ValueError(
                f"Unexpected shape of provided {parameter_name}!\n"
                f"Should be (length of dataset, dimensions={self.dimensions}) but is {binned_input.shape}"
            )

    @property
    def base_data(self) -> BaseDataContainer:
        raise NotImplementedError(
            "Base data is not available for BinnedDistributionFromHistogram which have been "
            "initialized from an already binned distribution.\n"
            "What you are trying to attempt is not possible."
        )

    @property
    def systematics(self) -> SystematicsInfo:
        raise NotImplementedError(
            "The systematics property is not available for "
            "BinnedDistributionFromHistogram which have been initialized from an already "
            "binned distribution.\nWhat you are trying to attempt is not possible."
        )

    @property
    def bin_covariance_matrix(self) -> np.ndarray:
        # TODO: Maybe make this also available for the BinnedDistributionFromHistogram variant!
        raise NotImplementedError(
            "The bin_covariance_matrix property is not available for "
            "BinnedDistributionFromHistogram which have been initialized from an already "
            "binned distribution.\nWhat you are trying to attempt is not possible."
        )

    @property
    def bin_correlation_matrix(self) -> np.ndarray:
        # TODO: Maybe make this also available for the BinnedDistributionFromHistogram variant!
        raise NotImplementedError(
            "The bin_correlation_matrix property is not available for "
            "BinnedDistributionFromHistogram which have been initialized from an already "
            "binned distribution.\nWhat you are trying to attempt is not possible."
        )

    def bin_errors_sq_with_normalization(
        self,
        normalization_factor: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        raise NotImplementedError(
            "The bin_errors_sq_with_normalization property is not available for "
            "BinnedDistributionFromHistogram which have been initialized from an already "
            "binned distribution.\nWhat you are trying to attempt is not possible."
        )
