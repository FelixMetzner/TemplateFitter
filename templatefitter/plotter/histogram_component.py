"""
Provides the HistComponent class, which is a container combining a BinnedDistribution with
information necessary to plot it, such as the label and color of the component in the plot.
"""

import numpy as np
import pandas as pd

from typing import Optional, Tuple

from templatefitter.binned_distributions.binning import Binning
from templatefitter.binned_distributions.weights import WeightsInputType
from templatefitter.binned_distributions.systematics import SystematicsInputType
from templatefitter.binned_distributions.binned_distribution import BinnedDistribution, DataInputType, \
    DataColumnNamesInput

__all__ = [
    "HistComponent"
]


class HistComponent:
    """
    Helper class for handling components of histograms.
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
    ):
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

        self._min_val = None
        self._max_val = None

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

    def get_underlying_binned_distribution(self, binning: Binning) -> BinnedDistribution:
        if self._binned_distribution is not None:
            if binning == self._binned_distribution.binning:
                return self._binned_distribution

        assert self.input_data is not None
        if isinstance(self.input_data, pd.DataFrame):
            assert self.input_column_name is not None

        binned_dist = BinnedDistribution(
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

    @property
    def raw_data_range(self) -> Tuple[float, float]:
        if self._min_val is not None and self._max_val is not None:
            return self._min_val, self._max_val

        if isinstance(self.input_data, pd.DataFrame) and self.input_column_name is None:
            raise RuntimeError("input_column_name must be defined if input_data is a pandas.DataFrame!")

        raw_data = BinnedDistribution.get_data_input(in_data=self.input_data, data_column_names=self.input_column_name)
        assert len(raw_data.shape) == 1, raw_data.shape

        self._min_val = np.amin(raw_data) if len(raw_data) > 0 else +float("inf")
        self._max_val = np.amax(raw_data) if len(raw_data) > 0 else -float("inf")

        return self._min_val, self._max_val

    @property
    def min_val(self) -> float:
        return self.raw_data_range[0]

    @property
    def max_val(self) -> float:
        return self.raw_data_range[1]

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
