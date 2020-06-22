"""
# TODO: This is not yet completed and unused!
Provides a container class SubsetsPlotter which holds plots of the same variable
for different subsets of a dataset, e.g. different bins, channels or similar.
"""

import os
import logging

from typing import NamedTuple, Any, Union, Optional, List, Dict

from templatefitter.plotter.histogram_variable import HistVariable
from templatefitter.plotter.histogram_plot_base import HistogramPlot

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "SubsetsPlotter",
]


# TODO: This is not yet completed and unused!
class PlotInfoContainer(NamedTuple):
    key: str
    hist_var: HistVariable
    title: str
    info_text: str
    style: Optional[str] = None
    text_h_pos: float = 0.66
    text_v_pos: float = 0.88
    legend_loc: Optional[Union[int, str]] = None
    legend_cols: Optional[int] = None
    y_scale: Optional[float] = None


# TODO: This is not yet completed and unused!
class SubsetsPlotter:
    def __init__(
            self,
            variable: HistVariable,
            plot_class: HistogramPlot,
            plot_infos: Optional[PlotInfoContainer],
            **kwargs
    ) -> None:
        self._variable = variable  # type: HistVariable
        self._plot_class = plot_class  # type: HistogramPlot

        self._plot_infos = plot_infos  # type: Optional[PlotInfoContainer]
        self._optional_arguments_dict = kwargs  # type: Dict[str, Any]

        self._subset_name_list = []  # type: List[str]
        self._subset_latex_labels = {}  # type: Dict[str, str]
        self._subset_histograms = {}  # type: Dict[str, str]

    def add_component(self) -> None:
        pass

    def create_plots(self) -> Optional[Dict[str, Union[str, os.PathLike]]]:
        pass

    @property
    def variable(self) -> HistVariable:
        return self._variable

    @property
    def plot_class(self) -> HistogramPlot:
        return self._plot_class

    @property
    def plot_class_name(self) -> str:
        return self._plot_class.__class__.__name__
