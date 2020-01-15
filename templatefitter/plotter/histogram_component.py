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
    DataInputType, DataColumnNamesInput

__all__ = [
    "HistComponent"
]


# TODO: Differentiate between component from binned and raw Data (the latter is the only option available currently)
class HistComponent(ABC):
    """
    Abstract base class for helper classes for handling components of histograms.
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
        HistComponent constructor.

        :param label: Component label for the histogram.
        :param data: Data to be plotted as histogram.
        :param weights: Weights for the events in data.
        :param systematics: Information about the systematics associated with the data.
        :param color: Color of the histogram component.
        :param alpha: Alpha value of the histogram component.
        """

        self._input_data = data
        self._input_weights = weights
        self._input_systematics = systematics
        self._input_column_name = self._get_data_column_name(data_column_names=data_column_names)

        self._label = label
        self._color = color
        self._alpha = alpha

        self._raw_data = None
        self._min_val = None
        self._max_val = None
        self._raw_weights = None

        self._binned_distribution = None

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

    def get_histogram_squared_bin_errors(self, binning: Binning, normalization_factor: Optional[float]) -> np.ndarray:
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
    def label(self) -> str:
        return self._label

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

    @property
    def input_column_name(self) -> Optional[str]:
        return self._input_column_name

    @input_column_name.setter
    def input_column_name(self, input_column_names: DataColumnNamesInput) -> None:
        if self._input_column_name is not None:
            raise RuntimeError("You are trying to overwrite the HistogramComponents input_column_names!")
        self._input_column_name = self._get_data_column_name(data_column_names=input_column_names)

    @property
    def color(self) -> Optional[str]:
        return self._color

    @color.setter
    def color(self, color: str) -> None:
        self._color = color

    @property
    def alpha(self) -> float:
        return self._alpha

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
    def raw_data_size(self) -> int:
        return self._get_raw_data().size

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
    def raw_weights(self) -> np.ndarray:
        return self._get_raw_weights()

    @property
    def raw_weights_sum(self) -> float:
        raw_weights_sum = np.sum(self._get_raw_weights())
        assert isinstance(raw_weights_sum, float)
        return raw_weights_sum

    @staticmethod
    def _get_data_column_name(data_column_names: DataColumnNamesInput) -> Optional[str]:
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
        HistComponent constructor.

        :param label: Component label for the histogram.
        :param data: Data to be plotted as histogram.
        :param weights: Weights for the events in data.
        :param systematics: Information about the systematics associated with the data.
        :param color: Color of the histogram component.
        :param alpha: Alpha value of the histogram component.
        """
        super().__init__(
            label=label,
            data=data,
            color=color,
            alpha=alpha
        )

        self._input_data = data
        self._input_weights = weights
        self._input_systematics = systematics
        self._input_column_name = self._get_data_column_name(data_column_names=data_column_names)


class HistComponentFromHistogram(HistComponent):
    """
    Helper class for handling components of histograms.
    This implementation of the HistComponent class fills the histogram from already histogrammed data.
    The input is thus only the bin counts and the errors on the bins.
    However, this approach does not allow for extensive modifications of the histogram, as it would be possible
    if the underlying data is available (e.g. arbitrary rebinning is not possible).
    """
    pass
