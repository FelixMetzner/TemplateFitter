"""
Contains base class for histogram plots: HistogramPlot
"""
import logging

from typing import Optional, Tuple
from abc import ABC, abstractmethod
from matplotlib import pyplot as plt

from templatefitter.binned_distributions.binning import Binning

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

    def add_component(self) -> None:
        # TODO
        pass

    @abstractmethod
    def plot_on(self) -> Tuple[plt.figure, plt.axis]:
        # TODO
        pass

    @property
    def binning(self) -> Optional[Binning]:
        return self._histogram_dict.common_binning

    @property
    def variable(self) -> HistVariable:
        return self._variable

    def reset_binning_to_use_raw_data_range(self) -> None:
        # TODO
        pass

    def apply_adaptive_binning(self) -> None:
        # TODO
        pass

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
