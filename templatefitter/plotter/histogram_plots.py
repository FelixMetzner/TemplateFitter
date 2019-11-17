"""
Provides several specialized histogram plot classes.
"""
import logging

from typing import Optional, Tuple
from matplotlib import pyplot as plt, figure, axes, axis

from templatefitter.plotter import plot_style
from templatefitter.plotter.histogram_plot_base import HistogramPlot

from templatefitter.binned_distributions.weights import WeightsInputType
from templatefitter.binned_distributions.systematics import SystematicsInputType
from templatefitter.binned_distributions.binned_distribution import DataInputType

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "SimpleHistogramPlot",
    "StackedHistogramPlot",
    "DataMCHistogramPlot"
]

plot_style.set_matplotlibrc_params()


class SimpleHistogramPlot(HistogramPlot):

    def add_component(
            self,
            label: str,
            histogram_key: str,
            data: DataInputType,
            weights: WeightsInputType = None,
            systematics: SystematicsInputType = None,
            hist_type: Optional[str] = None,
            color: Optional[str] = None,
            alpha: float = 1.0
    ) -> None:
        # TODO: Ensure, that the histogram keys are all different, or remove histogram_key from signature, and just
        #       use a counter...
        self._add_component(
            label=label,
            histogram_key=histogram_key,
            data=data,
            weights=weights,
            systematics=systematics,
            hist_type=hist_type,
            color=color,
            alpha=alpha
        )

    def plot_on(
            self,
            ax: Optional[axis.Axis] = None,
            draw_legend: bool = True,
            legend_inside: bool = True,
            y_axis_scale: float = 1.3,
            normed: bool = False,
            y_label: str = "Events"
    ) -> Tuple[figure.Figure, axis.Axis]:
        if ax is None:
            _, ax = plt.subplots()
        self._last_figure = ax.get_figure()

        for histogram in self._histograms.histograms:
            assert histogram.number_of_components == 1, histogram.number_of_components
            ax.hist(
                x=histogram.get_bin_count_of_component(index=0),
                bins=self.bin_edges,
                density=normed,
                weights=histogram.get_component(index=0).weights,
                histtype=histogram.hist_type,
                label=histogram.get_component(index=0).label,
                alpha=histogram.get_component(index=0).alpha,
                lw=1.5,
                color=histogram.get_component(index=0).color
            )

        ax.set_xlabel(self.variable.x_label, plot_style.xlabel_pos)
        ax.set_ylabel(self._get_y_label(normed=normed, evts_or_cands=y_label), plot_style.ylabel_pos)

        # TODO: Maybe generalize the legend setup in a function and move to base class...
        if draw_legend:
            if legend_inside:
                ax.legend(frameon=False)
                y_limits = ax.get_ylim()
                ax.set_ylim(y_limits[0], y_axis_scale * y_limits[1])
            else:
                ax.legend(frameon=False, bbox_to_anchor=(1, 1))

        return ax


class StackedHistogramPlot(HistogramPlot):
    pass


class DataMCHistogramPlot(HistogramPlot):
    pass

    @staticmethod
    def create_hist_ratio_figure(
            fig_size: Tuple[float, float] = (5, 5),
            height_ratio: Tuple[float, float] = (3.5, 1)
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
