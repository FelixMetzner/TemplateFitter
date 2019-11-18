"""
Provides several specialized histogram plot classes.
"""
import copy
import logging
import numpy as np

import matplotlib.axes._axes as axes
from typing import Optional, Union, Tuple
from matplotlib import pyplot as plt, figure
from uncertainties import unumpy as unp, ufloat

from templatefitter.plotter import plot_style
from templatefitter.plotter.histogram_variable import HistVariable
from templatefitter.plotter.histogram_plot_base import HistogramPlot

from templatefitter.binned_distributions.weights import WeightsInputType
from templatefitter.binned_distributions.systematics import SystematicsInputType
from templatefitter.binned_distributions.binned_distribution import DataInputType

from templatefitter.stats import pearson_chi2_test, cowan_binned_likelihood_gof, toy_chi2_test

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "SimpleHistogramPlot",
    "StackedHistogramPlot",
    "DataMCHistogramPlot"
]

plot_style.set_matplotlibrc_params()


class SimpleHistogramPlot(HistogramPlot):
    def __init__(self, variable: HistVariable):
        super().__init__(variable=variable)

    def add_component(
            self,
            label: str,
            data: DataInputType,
            weights: WeightsInputType = None,
            hist_type: Optional[str] = None,
            color: Optional[str] = None,
            alpha: float = 1.0
    ) -> None:
        auto_histogram_key = f"hist_{self.number_of_histograms}"
        self._add_component(
            label=label,
            histogram_key=auto_histogram_key,
            data=data,
            weights=weights,
            systematics=None,
            hist_type=hist_type,
            color=color,
            alpha=alpha
        )

    def plot_on(
            self,
            ax: Optional[axes.Axes] = None,
            draw_legend: bool = True,
            legend_inside: bool = True,
            y_axis_scale: float = 1.3,
            normed: bool = False,
            y_label: str = "Events"
    ) -> axes.Axes:
        if ax is None:
            _, ax = plt.subplots()
        self._last_figure = ax.get_figure()

        for histogram in self._histograms.histograms:
            assert histogram.number_of_components == 1, histogram.number_of_components
            ax.hist(
                x=np.ones(histogram.binning.num_bins[0]),
                bins=self.bin_edges,
                density=normed,
                weights=histogram.get_bin_count_of_component(index=0),
                histtype=histogram.hist_type,
                label=histogram.get_component(index=0).label,
                alpha=histogram.get_component(index=0).alpha,
                lw=1.5,
                color=histogram.get_component(index=0).color
            )

        ax.set_xlabel(self.variable.x_label, plot_style.xlabel_pos)
        ax.set_ylabel(self._get_y_label(normed=normed, evts_or_cands=y_label), plot_style.ylabel_pos)

        if draw_legend:
            if legend_inside:
                ax.legend(frameon=False)
                y_limits = ax.get_ylim()
                ax.set_ylim(bottom=y_limits[0], top=y_axis_scale * y_limits[1])
            else:
                ax.legend(frameon=False, bbox_to_anchor=(1, 1))

        return ax


class StackedHistogramPlot(HistogramPlot):
    def __init__(self, variable: HistVariable):
        super().__init__(variable=variable)

    def add_component(
            self,
            label: str,
            data: DataInputType,
            weights: WeightsInputType = None,
            color: Optional[str] = None,
            alpha: float = 1.0
    ) -> None:
        self._add_component(
            label=label,
            histogram_key="stacked_histogram",
            data=data,
            weights=weights,
            systematics=None,
            hist_type="stepfilled",
            color=color,
            alpha=alpha
        )

    def plot_on(
            self,
            ax: Optional[axes.Axes] = None,
            draw_legend: bool = True,
            legend_inside: bool = True,
            y_axis_scale: float = 1.6,
            y_label: str = "Events"
    ) -> axes.Axes:

        if ax is None:
            _, ax = plt.subplots()
        self._last_figure = ax.get_figure()

        histogram = self._histograms.histograms[0]

        ax.hist(
            x=[np.ones(len(bin_count)) for bin_count in histogram.get_bin_counts],
            bins=self.bin_edges,
            weights=histogram.get_bin_counts,
            stacked=True,
            edgecolor="black",
            lw=0.3,
            color=histogram.colors,
            label=histogram.labels,
            histtype=histogram.hist_type
        )

        ax.set_xlabel(self._variable.x_label, plot_style.xlabel_pos)
        ax.set_ylabel(self._get_y_label(normed=False, evts_or_cands=y_label), plot_style.ylabel_pos)

        if draw_legend:
            if legend_inside:
                ax.legend(frameon=False)
                y_limits = ax.get_ylim()
                ax.set_ylim(bottom=y_limits[0], top=y_axis_scale * y_limits[1])
            else:
                ax.legend(frameon=False, bbox_to_anchor=(1, 1))

        return ax


class DataMCHistogramPlot(HistogramPlot):
    valid_styles = ["stacked", "summed"]
    valid_ratio_types = ["normal", "vs_uncert"]
    valid_gof_methods = ["pearson", "cowan", "toys"]

    data_key = "data_histogram"
    mc_key = "mc_histogram"

    def __init__(self, variable: HistVariable):
        super().__init__(variable=variable)

    def add_data_component(
            self,
            label: str,
            data: DataInputType,
            color: str = plot_style.KITColors.kit_black
    ) -> None:
        if self.data_key in self._histograms.histogram_keys:
            raise RuntimeError(f"A data component has already been added to {self.__class__.__name__} instance!")
        self._add_component(
            label=label,
            histogram_key=self.data_key,
            data=data,
            weights=None,
            systematics=None,
            hist_type="stepfilled",  # TODO: Define new own hist_type for data plots!
            color=color,
            alpha=1.0
        )

    def add_mc_component(
            self,
            label: str,
            data: DataInputType,
            weights: WeightsInputType = None,
            systematics: SystematicsInputType = None,
            color: Optional[str] = None
    ) -> None:
        self._add_component(
            label=label,
            histogram_key=self.mc_key,
            data=data,
            weights=weights,
            systematics=systematics,
            hist_type="stepfilled",
            color=color,
            alpha=1.0
        )

    # Same as add_mc_component method; just added to satisfy requirements from base class.
    def add_component(
            self,
            label: str,
            data: DataInputType,
            weights: WeightsInputType = None,
            systematics: SystematicsInputType = None,
            color: Optional[str] = None
    ) -> None:
        self._add_component(
            label=label,
            histogram_key=self.mc_key,
            data=data,
            weights=weights,
            systematics=systematics,
            hist_type="stepfilled",
            color=color,
            alpha=1.0
        )

    def plot_on(
            self,
            ax1: Optional[axes.Axes] = None,
            ax2: Optional[axes.Axes] = None,
            normalize_to_data: bool = True,
            style: str = "stacked",
            ratio_type: str = "normal",
            gof_check_method: Optional[str] = None,
            include_sys: bool = False,
            y_label: str = "Events",
            sum_color: str = plot_style.KITColors.kit_purple,
            draw_legend: bool = True,
            legend_inside: bool = True,
            legend_cols: Optional[int] = None,
            legend_loc: Optional[Union[int, str]] = None,
            markers_with_width: bool = True,
            plot_outlier_indicators: bool = True,
            adaptive_binning: bool = False,
            y_scale: float = 1.1
    ) -> Tuple[float, int, float, float]:
        ax1, ax2 = self._check_axes_input(ax1=ax1, ax2=ax2)
        self._last_figure = ax1.get_figure()

        self._check_style_settings_input(style=style, ratio_type=ratio_type, gof_check_method=gof_check_method)

        plot_outlier_indicators = self._check_outlier_indicator_setting(
            outlier_indicator_setting=plot_outlier_indicators,
            ratio_type=ratio_type
        )

        if legend_cols is None:
            legend_cols = 2
        if legend_loc is None:
            legend_loc = 0

        if adaptive_binning:
            self._histograms.apply_adaptive_binning_based_on_key(
                key=self.mc_key,
                minimal_bin_count=5,
                minimal_number_of_bins=7
            )

        bin_scaling = 1. / np.around(self.bin_widths / self.minimal_bin_width, decimals=0)

        sum_w = np.sum(np.array(self._histograms[self.mc_key].get_bin_counts()), axis=0)

        if normalize_to_data:
            norm_factor = self._histograms[self.data_key].raw_data_size / self._histograms[self.mc_key].raw_weight_sum
            sum_w = sum_w * norm_factor
        else:
            norm_factor = 1.

        sum_w2_stat_only = np.sum(np.array([np.histogram(
            mc_component.raw_data,
            bins=self.bin_edges,
            weights=np.square(mc_component.raw_weights * norm_factor)
        )[0] for mc_component in self._histograms[self.mc_key].components]), axis=0)

        if include_sys:
            sys_uncert2 = self._histograms[self.mc_key].get_systematic_uncertainty_per_bin()
            if sys_uncert2 is not None:
                sum_w2 = sum_w2_stat_only + sys_uncert2
            else:
                sum_w2 = sum_w2_stat_only
        else:
            sum_w2 = sum_w2_stat_only

        data_bin_count = self._histograms[self.data_key].get_bin_counts()

        if style.lower() == "stacked":
            ax1.hist(
                x=[self.bin_mids for _ in range(self._histograms[self.mc_key].number_of_components)],
                bins=self.bin_edges,
                weights=self._histograms[self.mc_key].get_bin_counts(),
                stacked=True,
                edgecolor="black",
                lw=0.3,
                color=self._histograms[self.mc_key].colors,
                label=self._histograms[self.mc_key].labels,
                histtype='stepfilled'
            )

            ax1.bar(
                x=self.bin_mids,
                height=2 * np.sqrt(sum_w2),
                width=self.bin_widths,
                bottom=sum_w * bin_scaling - np.sqrt(sum_w2),
                color="black",
                hatch="///////",
                fill=False,
                lw=0,
                label="MC stat. unc." if not include_sys else "MC stat. + sys. unc."
            )
        elif style.lower() == "summed":
            ax1.bar(
                x=self.bin_mids,
                height=2 * np.sqrt(sum_w2),
                width=self.bin_widths,
                bottom=sum_w * bin_scaling - np.sqrt(sum_w2),
                color=sum_color,
                lw=0,
                label="MC" if not normalize_to_data else r"MC $\times$ " + f"{norm_factor:.2f}"
            )
        else:
            raise RuntimeError(f"Invalid style '{style.lower()}!'\n style must be one of {self.valid_styles}!")

        ax1.errorbar(
            x=self.bin_mids,
            y=data_bin_count * bin_scaling,
            yerr=np.sqrt(data_bin_count),
            xerr=self.bin_widths / 2 if markers_with_width else None,
            ls="",
            marker=".",
            color="black",
            label=self._histograms[self.data_key].labels[0]
        )

        ax1.set_ylabel(self._get_y_label(normed=False, evts_or_cands=y_label), plot_style.ylabel_pos)

        toy_output = None
        dof = self.number_of_bins - 1 if normalize_to_data else self.number_of_bins
        if gof_check_method is not None and gof_check_method.lower() == "pearson":
            chi2, ndf, p_val = pearson_chi2_test(data=data_bin_count, expectation=sum_w, dof=dof)
        elif gof_check_method is not None and gof_check_method.lower() == "cowan":
            chi2, ndf, p_val = cowan_binned_likelihood_gof(data=data_bin_count, expectation=sum_w, dof=dof)
        elif gof_check_method is not None and gof_check_method.lower() == "toys":
            chi2, p_val, toy_output = toy_chi2_test(
                data=data_bin_count,
                expectation=sum_w,
                error=data_bin_count,
                mc_cov=self._histograms[self.mc_key].get_covariance_matrix()
            )
            ndf = dof
        else:
            chi2, ndf, p_val = (None, None, None)

        if draw_legend:
            if legend_inside:
                if style == "stacked":
                    ax1.legend(frameon=False, ncol=legend_cols, loc=legend_loc, fontsize="smaller")
                else:
                    ax1.legend(frameon=False, ncol=legend_cols, loc=legend_loc)
            else:
                ax1.legend(frameon=False, bbox_to_anchor=(1, 1))

        y_limits = ax1.get_ylim()
        ax1.set_ylim(bottom=0., top=y_scale * y_limits[1])

        ax2.set_xlabel(self._variable.x_label, plot_style.xlabel_pos)

        if ratio_type.lower() == "normal":
            ax2.set_ylabel(r"$\frac{\mathrm{Data - MC}}{\mathrm{Data}}$")
            ax2.set_ylim(bottom=-1., top=1.)
        if ratio_type.lower() == "vs_uncert":
            if include_sys:
                ax2.set_ylabel(r"$\frac{\mathrm{Data - MC}}{\sigma_\mathrm{stat + sys}^\mathrm{Data - MC}}$")
            else:
                ax2.set_ylabel(r"$\frac{\mathrm{Data - MC}}{\sigma_\mathrm{stat}^\mathrm{Data - MC}}$")

        try:
            uh_data = unp.uarray(data_bin_count, np.sqrt(data_bin_count))
            uh_mc = unp.uarray(sum_w, np.sqrt(sum_w2))

            if ratio_type.lower() == "normal":
                divisor = copy.deepcopy(uh_data)
                divisor[divisor == 0] = ufloat(0.01, 0.1)
            elif ratio_type.lower() == "vs_uncert":
                divisor = unp.uarray(unp.std_devs(uh_data - uh_mc), 0.)
                divisor[divisor == 0] = ufloat(0.01, 0.1)
            else:
                divisor = None

            ratio = (uh_data - uh_mc) / divisor

            ratio[(uh_data == 0.) & (uh_mc == 0.)] = ufloat(0., 0.)

            if ratio_type.lower() == "normal":
                ratio[np.logical_xor((uh_data == 0.), (uh_mc == 0.))] = ufloat(-99, 0.)

            if ratio_type.lower() == "vs_uncert":
                max_val_mask = (uh_data != 0.) & (uh_mc != 0.) & ((uh_data - uh_mc) != 0)
                max_val = np.around(max(
                    abs(np.min(unp.nominal_values(ratio[max_val_mask]) - unp.std_devs(ratio[max_val_mask]))),
                    abs(np.max(unp.nominal_values(ratio[max_val_mask]) + unp.std_devs(ratio[max_val_mask])))
                ), decimals=1)
                assert isinstance(max_val, float), (type(max_val), max_val)
                ax2.set_ylim(bottom=-1. * max_val, top=max_val)
            else:
                max_val = 1.

            ax2.axhline(y=0, color=plot_style.KITColors.dark_grey, alpha=0.8)
            ax2.errorbar(
                self.bin_mids,
                unp.nominal_values(ratio),
                yerr=unp.std_devs(ratio),
                xerr=self.bin_widths / 2 if markers_with_width else None,
                ls="",
                marker=".",
                color=plot_style.KITColors.kit_black
            )

            for bin_mid, r_val, mc_val, data_val in zip(self.bin_mids, ratio, uh_mc, uh_data):
                if mc_val == 0. and ((data_val != 0. and ratio_type.lower() != "vs_uncert")
                                     or (abs(r_val) > max_val and ratio_type.lower() == "vs_uncert")):
                    ax2.text(x=bin_mid, y=+0.1 * max_val, s="No MC", fontsize=5, rotation=90, ha="center", va="bottom")
                    ax2.text(x=bin_mid, y=+0.1 * max_val, s=f"#Data={int(unp.nominal_values(data_val))}", fontsize=5,
                             rotation=90, ha="center", va="bottom")
                elif data_val == 0. and ((mc_val != 0. and ratio_type.lower() != "vs_uncert")
                                         or (abs(r_val) > max_val and ratio_type.lower() == "vs_uncert")):
                    ax2.text(x=bin_mid, y=+0.1 * max_val, s=f"#MC={unp.nominal_values(mc_val):.0f}", fontsize=5,
                             rotation=90, ha="center", va="bottom")
                    ax2.text(x=bin_mid, y=-0.1 * max_val, s="No Data", fontsize=5, rotation=90, ha="center", va="top")
                elif r_val > 1.0 and plot_outlier_indicators:
                    ax2.text(x=bin_mid, y=+0.08 * max_val, s=f"{unp.nominal_values(r_val):3.2f}" + r"$\rightarrow$",
                             fontsize=5, rotation=90, ha="right", va="bottom")
                elif r_val < -1.0 and plot_outlier_indicators:
                    ax2.text(x=bin_mid, y=-0.08 * max_val, s=r"$\leftarrow$" + f"{unp.nominal_values(r_val):3.2f}",
                             fontsize=5, rotation=90, ha="right", va="top")
                else:
                    pass

        except ZeroDivisionError:
            ax2.text(x=self.bin_mids[int(np.ceil(len(self.bin_mids) / 2.))], y=0.1,
                     s="DataMCHistogramPlot: ZeroDivisionError occurred!", fontsize=8, ha="center", va="bottom")
            ax2.axhline(y=0, color=plot_style.KITColors.dark_grey, alpha=0.8)

        plt.subplots_adjust(hspace=0.08)

        return chi2, ndf, p_val, toy_output

    @staticmethod
    def _check_style_settings_input(style: str, ratio_type: str, gof_check_method: Optional[str]) -> None:
        if not style.lower() in DataMCHistogramPlot.valid_styles:
            raise ValueError(f"The argument 'style' must be one of {DataMCHistogramPlot.valid_styles}!"
                             f"Provided was '{style}'")
        if not ratio_type.lower() in DataMCHistogramPlot.valid_ratio_types:
            raise ValueError(f"The argument 'ratio_type' must be one of {DataMCHistogramPlot.valid_ratio_types}!"
                             f"Provided was '{ratio_type}'")
        if gof_check_method is not None and gof_check_method.lower() not in DataMCHistogramPlot.valid_gof_methods:
            raise ValueError(f"The argument 'gof_check_method' must be one of {DataMCHistogramPlot.valid_gof_methods} "
                             f"or None! Provided was '{gof_check_method}'")

    @staticmethod
    def _check_outlier_indicator_setting(outlier_indicator_setting: bool, ratio_type: str) -> bool:
        if ratio_type.lower() == "vs_uncert":
            former_outlier_setting = outlier_indicator_setting
            outlier_indicator_setting = False
            if former_outlier_setting != outlier_indicator_setting:
                logging.info(f"Resetting 'plot_outlier_indicators' from True to False, "
                             f"because of 'ratio_type' being set to '{ratio_type.lower()}'.")
        return outlier_indicator_setting

    @staticmethod
    def _check_axes_input(ax1: axes.Axes, ax2: axes.Axes) -> Tuple[axes.Axes, axes.Axes]:
        if ax1 is None and ax2 is None:
            _, (ax1, ax2) = DataMCHistogramPlot.create_hist_ratio_figure()
            return ax1, ax2
        elif ax1 is not None and ax2 is not None:
            return ax1, ax2
        else:
            raise ValueError("Either specify both axes or leave both empty!")

    @staticmethod
    def create_hist_ratio_figure(
            fig_size: Tuple[float, float] = (5, 5),
            height_ratio: Tuple[float, float] = (3.5, 1)
    ) -> Tuple[figure.Figure, Tuple[axes.Axes, axes.Axes]]:
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
