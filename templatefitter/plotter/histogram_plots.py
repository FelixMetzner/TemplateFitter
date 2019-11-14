"""
Contains base class for histogram plots: HistogramPlot
"""
import logging

from templatefitter.plotter import plot_style
from templatefitter.plotter.histogram import Histogram, HistogramContainer

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "HistogramPlot"
]

plot_style.set_matplotlibrc_params()


class HistogramPlot:
    """
    Base class for histogram plots.
    """

    def __init__(self):
        self._histogram_dict = HistogramContainer()

    # TODO: WIP
