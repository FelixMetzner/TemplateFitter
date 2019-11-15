"""
Provides several specialized histogram plot classes.
"""
import logging

from typing import Tuple
from matplotlib import pyplot as plt, axes, figure


from templatefitter.plotter import plot_style
from templatefitter.plotter.histogram_plot_base import HistogramPlot

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "SimpleHistogramPlot",
    "StackedHistogramPlot",
    "DataMCHistogramPlot"
]

plot_style.set_matplotlibrc_params()


class SimpleHistogramPlot(HistogramPlot):
    pass


class StackedHistogramPlot(HistogramPlot):
    pass


class DataMCHistogramPlot(HistogramPlot):
    pass

    @staticmethod
    def create_hist_ratio_figure(
            fig_size=(5, 5),
            height_ratio=(3.5, 1)
    ) -> Tuple[figure.Figure, axes.Axes]:
        """
        Create a matplotlib.Figure for histogram ratio plots.

        :param fig_size: Size of full figure. Default is (5, 5).
        :param height_ratio: Size of main plot vs. size of ratio plot. Default is (3.5, 1).
        :return: A matplotlib.figure.Figure instance and a matplotlib.axes.Axes instance containing two axis.
        """
        fig, axs = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=fig_size,
            dpi=200,
            sharex='none',
            gridspec_kw={"height_ratios": [height_ratio[0], height_ratio[1]]}
        )

        assert isinstance(fig, figure.Figure), type(fig)
        assert len(axs) == 2, (len(axs), axs)

        return fig, axs
