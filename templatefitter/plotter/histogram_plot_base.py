"""
Contains abstract base class for histogram plots --- HistogramPlot.
"""
import logging
import numpy as np

from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
from typing import Optional, Union, Any, Tuple

from templatefitter.utility import PathType
from templatefitter.binned_distributions.binning import Binning
from templatefitter.binned_distributions.weights import WeightsInputType
from templatefitter.binned_distributions.systematics import SystematicsInputType
from templatefitter.binned_distributions.binned_distribution import DataInputType, DataColumnNamesInput

from templatefitter.plotter import plot_style
from templatefitter.plotter.histogram_variable import HistVariable
from templatefitter.plotter.plot_utilities import AxesType, FigureType
from templatefitter.plotter.histogram import Histogram, HistogramContainer

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "HistogramPlot",
]

plot_style.set_matplotlibrc_params()


class HistogramPlot(ABC):
    """
    Base class for histogram plots.
    """

    legend_cols_default = 1
    legend_loc_default = plt.rcParams["legend.loc"]
    legend_font_size_default = plt.rcParams["legend.fontsize"]

    def __init__(self, variable: HistVariable) -> None:
        self._variable = variable  # type: HistVariable
        self._histograms = HistogramContainer()

        self._last_figure = None  # type: Optional[FigureType]

    @abstractmethod
    def plot_on(self, *args, **kwargs) -> Union[AxesType, Tuple[FigureType, Tuple[AxesType, AxesType]], Any]:
        raise NotImplementedError(f"The 'plot_on' method is not implemented for the class {self.__class__.__name__}!")

    @abstractmethod
    def add_component(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            f"The 'add_component' method is not implemented for the class {self.__class__.__name__}!"
        )

    def _add_component(
        self,
        label: str,
        histogram_key: str,
        data: DataInputType,
        weights: WeightsInputType = None,
        systematics: SystematicsInputType = None,
        hist_type: Optional[str] = None,
        color: Optional[str] = None,
        alpha: float = 1.0,
    ) -> None:
        if histogram_key not in self._histograms.histogram_keys:
            new_histogram = Histogram(variable=self.variable, hist_type=hist_type)
            self._histograms.add_histogram(key=histogram_key, histogram=new_histogram)

        self._histograms[histogram_key].add_histogram_component(
            label=label,
            data=data,
            weights=weights,
            systematics=systematics,
            data_column_names=self.variable.df_label,
            color=color,
            alpha=alpha
        )

    def _add_prebinned_component(
        self,
        label: str,
        histogram_key: str,
        bin_counts: np.ndarray,
        original_binning: Binning,
        bin_errors_squared: np.ndarray = None,
        data_column_names: DataColumnNamesInput = None,
        hist_type: Optional[str] = None,
        color: Optional[str] = None,
        alpha: float = 1.0,
    ) -> None:
        if histogram_key not in self._histograms.histogram_keys:
            new_histogram = Histogram(variable=self.variable, hist_type=hist_type)
            self._histograms.add_histogram(key=histogram_key, histogram=new_histogram)

        if data_column_names is not None:
            if isinstance(data_column_names, str):
                assert data_column_names == self.variable.df_label, (data_column_names, self.variable.df_label)
            elif isinstance(data_column_names, list):
                assert len(data_column_names) == 1, (len(data_column_names), data_column_names)
                assert data_column_names[0] == self.variable.df_label, (data_column_names[0], self.variable.df_label)
            else:
                raise RuntimeError(f"Unexpected type for argument 'data_column_names':\n"
                                   f"Should be None, str or List[str], but is {type(data_column_names)}!")

        self._histograms[histogram_key].add_histogram_component(
            label=label,
            bin_counts=bin_counts,
            original_binning=original_binning,
            bin_errors_squared=bin_errors_squared,
            data_column_names=self.variable.df_label,
            color=color,
            alpha=alpha,
        )

    @property
    def binning(self) -> Optional[Binning]:
        return self._histograms.common_binning

    @property
    def bin_edges(self) -> Tuple[float, ...]:
        assert len(self.binning.bin_edges) == 1, self.binning.bin_edges
        return self.binning.bin_edges[0]

    @property
    def bin_widths(self) -> np.ndarray:
        assert len(self.binning.bin_widths) == 1, self.binning.bin_widths
        return np.array(self.binning.bin_widths[0])

    @property
    def bin_mids(self) -> Tuple[float, ...]:
        assert len(self.binning.bin_mids) == 1, self.binning.bin_mids
        return self.binning.bin_mids[0]

    @property
    def number_of_bins(self) -> int:
        assert len(self.binning.num_bins) == 1, self.binning.num_bins
        return self.binning.num_bins[0]

    @property
    def minimal_bin_width(self) -> float:
        return min(self.bin_widths)

    @property
    def variable(self) -> HistVariable:
        return self._variable

    @property
    def number_of_histograms(self) -> int:
        return self._histograms.number_of_histograms

    def reset_binning_to_use_raw_data_range(self) -> None:
        self._histograms.reset_binning_to_use_raw_data_range_of_all()

    def reset_binning_to_use_raw_data_range_of_histogram(self, histogram_key: str) -> None:
        self._histograms.reset_binning_to_use_raw_data_range_of_key(key=histogram_key)

    def apply_adaptive_binning_based_on_histogram(
        self,
        histogram_key: str,
        minimal_bin_count: int = 5,
        minimal_number_of_bins: int = 7,
    ) -> None:
        self._histograms.apply_adaptive_binning_based_on_key(
            key=histogram_key,
            minimal_bin_count=minimal_bin_count,
            minimal_number_of_bins=minimal_number_of_bins,
        )

    def _get_y_label(self, normed: bool, evts_or_cands: str = "Events") -> str:
        if normed:
            return "Normalized in arb. units"
        elif self._variable.use_log_scale:
            return f"{evts_or_cands} / Bin"
        else:
            return "{e} / {bo}{b:.2g}{v}{bc}".format(
                e=evts_or_cands,
                b=self.minimal_bin_width,
                v=" " + self._variable.unit if self._variable.unit else "",
                bo="(" if self._variable.unit else "",
                bc=")" if self._variable.unit else "",
            )

    def draw_legend(
        self,
        axis: AxesType,
        inside: bool,
        loc: Optional[Union[int, str]] = None,
        ncols: Optional[int] = None,
        y_axis_scale: Optional[float] = None,
        font_size: Optional[Union[int, float, str]] = None,
        bbox_to_anchor_tuple: Tuple[float, float] = None,
    ) -> None:
        if loc is None:
            loc = self.legend_loc_default
        if ncols is None:
            ncols = self.legend_cols_default
        if font_size is None:
            font_size = self.legend_font_size_default

        if inside:
            axis.legend(frameon=False, loc=loc, ncol=ncols, fontsize=font_size)

            if y_axis_scale is not None:
                y_limits = axis.get_ylim()
                axis.set_ylim(bottom=y_limits[0], top=y_axis_scale * y_limits[1])
        else:
            if bbox_to_anchor_tuple is None:
                bbox_to_anchor_tuple = (1., 1.)

            axis.legend(frameon=False, loc=loc, ncol=ncols, bbox_to_anchor=bbox_to_anchor_tuple)

    def get_last_figure(self) -> Optional[FigureType]:
        return self._last_figure

    def write_hist_data_to_file(self, file_path: PathType):
        self._histograms.write_to_file(file_path=file_path)
