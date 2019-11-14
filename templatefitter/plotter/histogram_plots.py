"""
# TODO
"""
import logging

from templatefitter.plotter import plot_style

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "HistogramPlot"
]

plot_style.set_matplotlibrc_params()


class HistogramPlot:
    """
    TODO
    """

    def __init__(self):
        self._histogram_dict = ...  # TODO
