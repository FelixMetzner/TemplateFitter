"""
Provides the HistComponent class, which is a container combining a BinnedDistribution with
information necessary to plot it, such as the label and color of the component in the plot.
"""
import numpy as np

from typing import Optional

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
            hist_type: Optional[str] = None,
            color: Optional[str] = None,
            alpha: float = 1.0
    ):
        """
        HistComponent constructor.

        :param label: Component label for the histogram.
        :param data: Data to be plotted as histogram.
        :param weights: Weights for the events in data.
        :param systematics: Information about the systematics associated with the data.
        :param hist_type: Specifies the histogram type of the component in the histogram.
        :param color: Color of the histogram component.
        :param alpha: Alpha value of the histogram component.
        """

        self._input_data = data
        self._input_weights = weights
        self._input_systematics = systematics
        self._input_column_names = data_column_names

        self._label = label
        self._hist_type = hist_type
        self._color = color
        self._alpha = alpha

        self._binned_distribution = None

    def get_histogram_bin_count(self, binning: Binning) -> np.ndarray:
        """
        Calculates the bin count for this component for a given binning.
        :param binning: The binning to be used to generate the histogram and calculate the bin count.
        :return: A np.ndarray containing the bin counts.
        """
        if self._binned_distribution is not None:
            assert isinstance(self._binned_distribution, BinnedDistribution), type(self._binned_distribution)
            if self._binned_distribution.binning == binning:
                bin_count = self._binned_distribution.bin_counts
                assert bin_count is not None
                return bin_count

        binned_dist = BinnedDistribution(
            bins=binning.bin_edges,
            dimensions=1,
            log_scale_mask=binning.log_scale_mask,
            data=self.input_data,
            weights=self.input_weights,
            systematics=self.input_systematics,
            data_column_names=self.input_column_names
        )

        assert isinstance(binned_dist, BinnedDistribution), type(binned_dist)
        assert not binned_dist.is_empty
        self._binned_distribution = binned_dist
        return binned_dist.bin_counts

    @property
    def label(self) -> str:
        return self._label

    @property
    def input_data(self) -> DataInputType:
        return self._input_data

    @property
    def input_weights(self) -> WeightsInputType:
        return self._input_weights

    @property
    def input_systematics(self) -> SystematicsInputType:
        return self._input_systematics

    @property
    def input_column_names(self) -> DataColumnNamesInput:
        return self._input_column_names

    @property
    def hist_type(self) -> Optional[str]:
        return self._hist_type

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
    def min_val(self) -> Optional[float]:
        if self._binned_distribution is None:
            return None
        histogram_range = self._binned_distribution.binning.range
        assert len(histogram_range) == 1, (len(histogram_range), len(histogram_range))
        assert len(histogram_range[0]) == 2, (len(histogram_range[0]), histogram_range[0])
        return histogram_range[0][0]

    @property
    def max_val(self) -> Optional[float]:
        if self._binned_distribution is None:
            return None
        histogram_range = self._binned_distribution.binning.range
        assert len(histogram_range) == 1, (len(histogram_range), len(histogram_range))
        assert len(histogram_range[0]) == 2, (len(histogram_range[0]), histogram_range[0])
        return histogram_range[0][1]
