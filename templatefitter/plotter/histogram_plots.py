"""
Contains base class for histogram plots: HistogramPlot
"""
import logging

from typing import Optional, Tuple
from abc import ABC, abstractmethod
from matplotlib import pyplot as plt

from templatefitter.binned_distributions.binning import Binning
from templatefitter.binned_distributions.weights import WeightsInputType
from templatefitter.binned_distributions.systematics import SystematicsInputType
from templatefitter.binned_distributions.binned_distribution import DataInputType, DataColumnNamesInput

from templatefitter.plotter import plot_style
from templatefitter.plotter.histogram import HistogramContainer
from templatefitter.plotter.histogram_variable import HistVariable

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "HistogramPlot"
]

plot_style.set_matplotlibrc_params()


class HistogramPlot(ABC):
    """
    Base class for histogram plots.
    """

    def __init__(self, variable: HistVariable):
        self._variable = variable  # type: HistVariable
        self._histogram_dict = HistogramContainer()

    def add_component(
            self,
            label: str,
            histogram_key: str,
            data: DataInputType,
            weights: WeightsInputType = None,
            systematics: SystematicsInputType = None,
            hist_type: str = 'step',
            color: Optional[str] = None,
            alpha: float = 1.0
    ) -> None:
        # TODO
        pass

    @abstractmethod
    def plot_on(self) -> Tuple[plt.figure, plt.axis]:
        raise NotImplementedError(f"The 'plot_on' method is not implemented for the class {self.__class__.__name__}!")

    @property
    def binning(self) -> Optional[Binning]:
        return self._histogram_dict.common_binning

    @property
    def variable(self) -> HistVariable:
        return self._variable

    def reset_binning_to_use_raw_data_range(self) -> None:
        self._histogram_dict.reset_binning_to_use_raw_data_range_of_all()

    def reset_binning_to_use_raw_data_range_of_histogram(self, histogram_key: str) -> None:
        self._histogram_dict.reset_binning_to_use_raw_data_range_of_key(key=histogram_key)

    def apply_adaptive_binning_based_on_histogram(
            self,
            histogram_key: str,
            minimal_bin_count: int = 5,
            minimal_number_of_bins: int = 7
    ) -> None:
        self._histogram_dict.apply_adaptive_binning_based_on_key(
            key=histogram_key,
            minimal_bin_count=minimal_bin_count,
            minimal_number_of_bins=minimal_number_of_bins
        )

    def _get_y_label(self, normed: bool, evts_or_cands: str = "Events") -> str:
        if normed:
            return "Normalized in arb. units"
        elif self._variable.use_log_scale:
            return f"{evts_or_cands} / Bin"
        else:
            bin_widths = self.binning.bin_widths[0]
            assert isinstance(bin_widths, tuple), (bin_widths, type(bin_widths))
            bin_width = min(bin_widths)
            return "{e} / ({b:.2g}{v})".format(
                e=evts_or_cands,
                b=bin_width,
                v=" " + self._variable.unit if self._variable.unit else ""
            )
