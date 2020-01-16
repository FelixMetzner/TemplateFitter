"""
Provides the HistComponent class, which is a container combining a BinnedDistribution with
information necessary to plot it, such as the label and color of the component in the plot.
"""

import numpy as np
import pandas as pd

from typing import Optional, Tuple
from abc import ABC, abstractmethod

from templatefitter.binned_distributions.binning import Binning
from templatefitter.binned_distributions.systematics import SystematicsInputType
from templatefitter.binned_distributions.weights import Weights, WeightsInputType
from templatefitter.binned_distributions.binned_distribution import BinnedDistribution, BinnedDistributionFromData, \
    BinnedDistributionFromHistogram, DataInputType, DataColumnNamesInput

__all__ = [
    "HistComponent",
    "HistComponentFromData",
    "HistComponentFromHistogram",
    "create_histogram_component"
]


class HistComponent(ABC):
    """
    Abstract base class for helper classes for handling components of histograms.
    """

    def __init__(
            self,
            label: str,
            data_column_names: DataColumnNamesInput = None,
            color: Optional[str] = None,
            alpha: float = 1.0
    ) -> None:
        """
        HistComponent constructor.

        :param label: Component label for the histogram.
        :param data_column_names: Optional string or list of strings with
                                  the column names of the variables to be plotted.
        :param color: Color of the histogram component.
        :param alpha: Alpha value of the histogram component.
        """

        self._input_data = None
        self._input_weights = None
        self._input_systematics = None
        self._input_column_name = self._get_data_column_names(data_column_names=data_column_names)

        self._label = label
        self._color = color
        self._alpha = alpha

        self._raw_data = None
        self._min_val = None
        self._max_val = None
        self._raw_weights = None

        self._binned_distribution = None

    @abstractmethod
    def get_histogram_bin_count(self, binning: Binning) -> np.ndarray:
        """
        Calculates the bin count for this component for a given binning.
        :param binning: The binning to be used to generate the histogram and calculate the bin count.
        :return: A np.ndarray containing the bin counts.
        """
        raise NotImplementedError("This method is not implemented for the abstract base class HistComponent, "
                                  "as it depends on the specific versions of the child classes.")

    @abstractmethod
    def get_histogram_squared_bin_errors(
            self,
            binning: Binning,
            normalization_factor: Optional[float] = None
    ) -> np.ndarray:
        raise NotImplementedError("This method is not implemented for the abstract base class HistComponent, "
                                  "as it depends on the specific versions of the child classes.")

    @abstractmethod
    def get_underlying_binned_distribution(self, binning: Binning) -> BinnedDistribution:
        raise NotImplementedError("This method is not implemented for the abstract base class HistComponent, "
                                  "as it depends on the specific versions of the child classes.")

    @property
    def label(self) -> str:
        return self._label

    @property
    def color(self) -> Optional[str]:
        return self._color

    @color.setter
    def color(self, color: str) -> None:
        self._color = color

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def input_column_name(self) -> Optional[str]:
        return self._input_column_name

    @input_column_name.setter
    def input_column_name(self, input_column_names: DataColumnNamesInput) -> None:
        if self._input_column_name is not None:
            raise RuntimeError("You are trying to overwrite the HistogramComponents input_column_names!")
        self._input_column_name = self._get_data_column_names(data_column_names=input_column_names)

    @property
    def raw_data(self) -> np.ndarray:
        raise NotImplementedError(f"This method is not implemented for the class {self.__class__.__name__}!")

    @property
    def raw_data_range(self) -> Tuple[float, float]:
        raise NotImplementedError(f"This method is not implemented for the class {self.__class__.__name__}!")

    @property
    @abstractmethod
    def min_val(self) -> float:
        raise NotImplementedError("This method is not implemented for the abstract base class HistComponent, "
                                  "as it depends on the specific versions of the child classes.")

    @property
    @abstractmethod
    def max_val(self) -> float:
        raise NotImplementedError("This method is not implemented for the abstract base class HistComponent, "
                                  "as it depends on the specific versions of the child classes.")

    @property
    def raw_data_size(self) -> int:
        raise NotImplementedError(f"This method is not implemented for the class {self.__class__.__name__}!")

    @property
    def raw_weights(self) -> np.ndarray:
        raise NotImplementedError(f"This method is not implemented for the class {self.__class__.__name__}!")

    @property
    def raw_weights_sum(self) -> float:
        raise NotImplementedError(f"This method is not implemented for the class {self.__class__.__name__}!")

    @staticmethod
    def _get_data_column_names(data_column_names: DataColumnNamesInput) -> Optional[str]:
        if isinstance(data_column_names, str):
            return data_column_names
        elif isinstance(data_column_names, list):
            assert len(data_column_names) == 1, (len(data_column_names), data_column_names)
            assert isinstance(data_column_names[0], str), data_column_names
            return data_column_names[0]
        elif data_column_names is None:
            return data_column_names
        else:
            raise TypeError(f"data_column_names must be either a string, a list containing ONE string, or None.\n"
                            f"You provided {data_column_names}")


class HistComponentFromData(HistComponent):
    """
    Helper class for handling components of histograms.
    This implementation of the HistComponent class fills the histogram from data and thus allows
    for rebinning and other more extensive modifications of the histogram.
    """

    def __init__(
            self,
            label: str,
            data: DataInputType,
            weights: WeightsInputType = None,
            systematics: SystematicsInputType = None,
            data_column_names: DataColumnNamesInput = None,
            color: Optional[str] = None,
            alpha: float = 1.0
    ) -> None:
        """
        HistComponentFromData constructor.

        :param label: Component label for the histogram.
        :param data: Data to be plotted as histogram.
        :param weights: Weights for the events in data.
        :param systematics: Information about the systematics associated with the data.
        :param data_column_names: Optional string or list of strings with
                                  the column names of the variables to be plotted.
        :param color: Color of the histogram component.
        :param alpha: Alpha value of the histogram component.
        """
        super().__init__(
            label=label,
            data_column_names=data_column_names,
            color=color,
            alpha=alpha
        )

        self._input_data = data
        self._input_weights = weights
        self._input_systematics = systematics

    def get_histogram_bin_count(self, binning: Binning) -> np.ndarray:
        """
        Calculates the bin count for this component for a given binning.
        :param binning: The binning to be used to generate the histogram and calculate the bin count.
        :return: A np.ndarray containing the bin counts.
        """
        binned_dist = self.get_underlying_binned_distribution(binning=binning)
        bin_count = binned_dist.bin_counts
        assert bin_count is not None
        return bin_count

    def get_histogram_squared_bin_errors(
            self,
            binning: Binning,
            normalization_factor: Optional[float] = None
    ) -> np.ndarray:
        binned_dist = self.get_underlying_binned_distribution(binning=binning)
        bin_errors_sq = binned_dist.bin_errors_sq_with_normalization(normalization_factor=normalization_factor)
        assert bin_errors_sq is not None
        return bin_errors_sq

    def get_underlying_binned_distribution(self, binning: Binning) -> BinnedDistribution:
        if self._binned_distribution is not None:
            if binning == self._binned_distribution.binning:
                return self._binned_distribution

        assert self.input_data is not None
        if isinstance(self.input_data, pd.DataFrame):
            assert self.input_column_name is not None

        binned_dist = BinnedDistributionFromData(
            bins=binning.bin_edges,
            dimensions=1,
            log_scale_mask=binning.log_scale_mask,
            data=self.input_data,
            weights=self.input_weights,
            systematics=self.input_systematics,
            data_column_names=self.input_column_name
        )

        assert isinstance(binned_dist, BinnedDistribution), type(binned_dist)
        assert not binned_dist.is_empty
        self._binned_distribution = binned_dist
        return binned_dist

    @property
    def input_data(self) -> DataInputType:
        return self._input_data

    @input_data.setter
    def input_data(self, input_data: DataInputType) -> None:
        if self._input_data is not None:
            raise RuntimeError("You are trying to overwrite the HistogramComponents input_data!")
        self._input_data = input_data

    @property
    def input_weights(self) -> WeightsInputType:
        return self._input_weights

    @input_weights.setter
    def input_weights(self, input_weights: WeightsInputType) -> None:
        if self._input_weights is not None:
            raise RuntimeError("You are trying to overwrite the HistogramComponents input_weights!")
        self._input_weights = input_weights

    @property
    def input_systematics(self) -> SystematicsInputType:
        return self._input_systematics

    @input_systematics.setter
    def input_systematics(self, input_systematics: SystematicsInputType) -> None:
        if self._input_systematics is not None:
            raise RuntimeError("You are trying to overwrite the HistogramComponents input_systematics!")
        self._input_systematics = input_systematics

    def _get_raw_data(self) -> np.ndarray:
        if self._raw_data is None:
            if isinstance(self.input_data, pd.DataFrame) and self.input_column_name is None:
                raise RuntimeError("input_column_name must be defined if input_data is a pandas.DataFrame!")
            raw_data = BinnedDistributionFromData.get_data_input(
                in_data=self.input_data,
                data_column_names=self.input_column_name
            )
            assert len(raw_data.shape) == 1, raw_data.shape
            self._raw_data = raw_data

        return self._raw_data

    @property
    def raw_data(self) -> np.ndarray:
        return self._get_raw_data()

    @property
    def raw_data_range(self) -> Tuple[float, float]:
        if self._min_val is not None and self._max_val is not None:
            return self._min_val, self._max_val

        raw_data = self._get_raw_data()

        self._min_val = np.amin(raw_data) if len(raw_data) > 0 else +float("inf")
        self._max_val = np.amax(raw_data) if len(raw_data) > 0 else -float("inf")

        return self._min_val, self._max_val

    @property
    def min_val(self) -> float:
        return self.raw_data_range[0]

    @property
    def max_val(self) -> float:
        return self.raw_data_range[1]

    def _get_raw_weights(self) -> np.ndarray:
        if self._raw_weights is None:
            raw_weights = Weights(
                weight_input=self._input_weights,
                data=self._get_raw_data(),
                data_input=self._input_data
            ).get_weights()
            assert len(raw_weights.shape) == 1, raw_weights.shape
            assert raw_weights.shape == self._get_raw_data().shape, (raw_weights.shape, self._get_raw_data().shape)
            self._raw_weights = raw_weights

        return self._raw_weights

    @property
    def raw_data_size(self) -> int:
        return self._get_raw_data().size

    @property
    def raw_weights(self) -> np.ndarray:
        return self._get_raw_weights()

    @property
    def raw_weights_sum(self) -> float:
        raw_weights_sum = np.sum(self._get_raw_weights())
        assert isinstance(raw_weights_sum, float)
        return raw_weights_sum


class HistComponentFromHistogram(HistComponent):
    """
    Helper class for handling components of histograms.
    This implementation of the HistComponent class fills the histogram from already histogrammed data.
    The input is thus only the bin counts and the errors on the bins.
    However, this approach does not allow for extensive modifications of the histogram, as it would be possible
    if the underlying data is available (e.g. arbitrary rebinning is not possible).
    """

    def __init__(
            self,
            label: str,
            bin_counts: DataInputType,
            original_binning: Binning,
            bin_errors_squared: np.ndarray = None,
            data_column_names: DataColumnNamesInput = None,
            color: Optional[str] = None,
            alpha: float = 1.0
    ) -> None:
        """
        HistComponentFromHistogram constructor.

        :param label: Component label for the histogram.
        :param bin_counts: Bin counts of the histogram to be plotted.
        :param bin_errors_squared: Squared errors for each bin of the histogram to be plotted.
        :param data_column_names: Optional string or list of strings with the (former)
                                  column names of the variables to be plotted.
        :param color: Color of the histogram component.
        :param alpha: Alpha value of the histogram component.
        """
        super().__init__(
            label=label,
            data_column_names=data_column_names,
            color=color,
            alpha=alpha
        )

        self._input_bin_counts = bin_counts
        self._input_bin_errors_squared = bin_errors_squared

        self._original_binning = original_binning
        if not self._original_binning.dimensions == 1:
            raise RuntimeError(f"A histogram component can only be one dimensional, "
                               f"but the provided binning is for {self._original_binning.dimensions} dimensions!")

        self._binned_distribution = BinnedDistributionFromHistogram(
            bins=self._original_binning.bin_edges,
            dimensions=self._original_binning.dimensions,
            scope=self._original_binning.range,
            log_scale_mask=self._original_binning.log_scale_mask,
            data=self._input_bin_counts,
            bin_errors_squared=self._input_bin_errors_squared
        )

    def get_histogram_bin_count(self, binning: Optional[Binning]) -> np.ndarray:
        """
        Returns the bin count for this component. The binning can be provided optionally, to check if.
        :param binning: The binning to be used to generate the histogram and calculate the bin count.
        :return: A np.ndarray containing the bin counts.
        """
        if binning is not None:
            if not binning == self._original_binning:
                raise RuntimeError("The provided binning does not agree with the original binning of the underlying "
                                   "BinnedDistributionFromHistogram:"
                                   "\nOriginal Binning\n\t" + "\n\t".join(self._original_binning.as_string_list)
                                   + "\nProvided Binning\n\t" + "\n\t".join(binning.as_string_list))
        return self._binned_distribution.bin_counts

    def get_histogram_squared_bin_errors(
            self,
            binning: Optional[Binning],
            normalization_factor: Optional[float] = None
    ) -> np.ndarray:
        if normalization_factor is not None:
            raise RuntimeError(f"The class {self.__class__.__name__} does not support the method "
                               f"get_histogram_squared_bin_errors with a normalization_factor other than 'None'!")

        if binning is not None:
            if not binning == self._original_binning:
                raise RuntimeError("The provided binning does not agree with the original binning of the underlying "
                                   "BinnedDistributionFromHistogram:"
                                   "\nOriginal Binning\n\t" + "\n\t".join(self._original_binning.as_string_list)
                                   + "\nProvided Binning\n\t" + "\n\t".join(binning.as_string_list))
        return self._binned_distribution.bin_errors_sq

    def get_underlying_binned_distribution(self, binning: Binning) -> BinnedDistribution:
        return self._binned_distribution

    @property
    def min_val(self) -> float:
        if self._min_val is not None:
            return self._min_val

        self._min_val = self._original_binning.range[0][0]
        return self._min_val

    @property
    def max_val(self) -> float:
        if self._max_val is not None:
            return self._max_val

        self._max_val = self._original_binning.range[0][1]
        return self._max_val

    @property
    def raw_data(self) -> np.ndarray:
        raise NotImplementedError(f"This method is not implemented for the class {self.__class__.__name__}!")

    @property
    def raw_data_range(self) -> Tuple[float, float]:
        raise NotImplementedError(f"This method is not implemented for the class {self.__class__.__name__}!")

    @property
    def raw_data_size(self) -> int:
        raise NotImplementedError(f"This method is not implemented for the class {self.__class__.__name__}!")

    @property
    def raw_weights(self) -> np.ndarray:
        raise NotImplementedError(f"This method is not implemented for the class {self.__class__.__name__}!")

    @property
    def raw_weights_sum(self) -> float:
        raise NotImplementedError(f"This method is not implemented for the class {self.__class__.__name__}!")


def create_histogram_component(*args, **kwargs) -> HistComponent:
    try:
        new_component = HistComponentFromData(*args, **kwargs)
    except TypeError:
        try:
            new_component = HistComponentFromHistogram(*args, **kwargs)
        except TypeError:
            raise TypeError("Failed to create a HistComponent from the provided input.\n"
                            "The input arguments must fit the signature of one of the HistComponent implementations:\n"
                            "\t 1) HistComponentFromData\n"
                            "\t 2) HistComponentFromHistogram\n"
                            "The provided input was:\n"
                            "\tTypes of args:\n\t\t- " + "\n\t\t- ".join([type(a) for a in args])
                            + "\n\tTypes of kwargs:\n\t\t- "
                            + "\n\t\t- ".join([f'{k}: {type(v)}' for k, v in kwargs.items()]))

    return new_component
