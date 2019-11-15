"""
Provides several specialized histogram plot classes.
"""
import logging

from typing import Optional, Tuple
from matplotlib import pyplot as plt, figure, axes, axis


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

    # TODO: WIP: Adapt this function!
    def plot_on(
            self,
            ax: Optional[axis.Axis] = None,
            draw_legend: bool = True,
            legend_inside: bool = True,
            y_axis_scale=1.3,
            normed: bool = False,
            y_label="Events"
    ) -> Tuple[figure.Figure, axis.Axis]:
        if ax is None:
            _, ax = plt.subplots()
        self._last_figure = ax.get_figure()

        bin_edges, bin_mids, bin_width = self._get_bin_edges()

        self._bin_edges = bin_edges
        self._bin_mids = bin_mids
        self._bin_width = bin_width

        for component in self._mc_components['single']:
            # if component.histtype == 'stepfilled':
            #     alpha = 0.9
            #     # edge_color = 'black'
            # else:
            #     # edge_color = None
            #     alpha = 1.0
            ax.hist(
                x=component.data,
                bins=bin_edges,
                density=normed,
                weights=component.weights,
                histtype=component.histtype,
                label=component.label,
                # edgecolor=edge_color if edge_color is not None else component.color,
                alpha=component.alpha,
                lw=1.5,
                color=component.color
            )

        ax.set_xlabel(self._variable.x_label, plot_style.xlabel_pos)

        y_label = self._get_y_label(normed=normed, bin_width=bin_width, evts_or_cand=ylabel)
        ax.set_ylabel(y_label, plot_style.ylabel_pos)

        if draw_legend:
            if legend_inside:
                ax.legend(frameon=False)
                ylims = ax.get_ylim()
                ax.set_ylim(ylims[0], yaxis_scale * ylims[1])
            else:
                ax.legend(frameon=False, bbox_to_anchor=(1, 1))

        return ax


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
