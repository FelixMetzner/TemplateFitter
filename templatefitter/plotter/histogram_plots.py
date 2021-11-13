"""
Provides several specialized histogram plot classes.
"""

import copy
import logging
import warnings
import numpy as np

from dataclasses import dataclass
from matplotlib import pyplot as plt, figure
from uncertainties import unumpy as unp, ufloat
from typing import Optional, Union, Tuple, List

from templatefitter.plotter import plot_style
from templatefitter.plotter.histogram import Histogram
from templatefitter.plotter.plot_utilities import AxesType
from templatefitter.plotter.histogram_variable import HistVariable
from templatefitter.plotter.histogram_plot_base import HistogramPlot

from templatefitter.binned_distributions.weights import WeightsInputType
from templatefitter.binned_distributions.binning import Binning, BinsInputType
from templatefitter.binned_distributions.systematics import SystematicsInputType
from templatefitter.binned_distributions.binned_distribution import DataInputType

from templatefitter.stats import pearson_chi2_test, cowan_binned_likelihood_gof, toy_chi2_test, ToyInfoOutputType

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "SimpleHistogramPlot",
    "StackedHistogramPlot",
    "DataMCHistogramPlot",
    "DataMCComparisonOutputType",
]

plot_style.set_matplotlibrc_params()


@dataclass(frozen=True)
class DataMCComparisonOutput:
    chi2: float
    ndf: int
    p_val: float
    test_method: str
    toy_output: Optional[ToyInfoOutputType]

    def __post_init__(self) -> None:
        if self.test_method not in self.valid_test_methods():
            raise ValueError(
                f"Argument test_method must be one of {self.valid_test_methods()} but is '{self.test_method}'!"
            )

    @staticmethod
    def valid_test_methods() -> Tuple[str, ...]:
        valid_test_method_names = ("pearson", "cowan", "toys", "toys_inverted")  # type: Tuple[str, ...]
        return valid_test_method_names

    @property
    def test_method_id(self) -> str:
        return self.test_method.capitalize()[0]


DataMCComparisonOutputType = Optional[DataMCComparisonOutput]


class SimpleHistogramPlot(HistogramPlot):
    def __init__(
        self,
        variable: HistVariable,
    ) -> None:
        super().__init__(variable=variable)

    def add_component(
        self,
        label: str,
        data: DataInputType,
        weights: WeightsInputType = None,
        hist_type: Optional[str] = None,
        color: Optional[str] = None,
        alpha: float = 1.0,
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
            alpha=alpha,
        )

    def plot_on(
        self,
        ax: Optional[AxesType] = None,
        draw_legend: bool = True,
        legend_inside: bool = True,
        y_axis_scale: float = 1.3,
        normed: bool = False,
        y_label: str = "Events",
    ) -> AxesType:
        if ax is None:
            _, ax = plt.subplots()
        self._last_figure = ax.get_figure()

        for histogram in self._histograms.histograms:
            assert histogram.number_of_components == 1, histogram.number_of_components
            if histogram.hist_type == "stepfilled":
                kwargs = {"lw": 0.3, "edgecolor": "black"}
            else:
                kwargs = {"lw": 1.5}
            ax.hist(
                x=histogram.binning.bin_mids[0],
                bins=self.bin_edges,
                density=normed,
                weights=histogram.get_bin_count_of_component(index=0),
                histtype=histogram.hist_type,
                label=histogram.get_component(index=0).label,
                alpha=histogram.get_component(index=0).alpha,
                color=histogram.get_component(index=0).color,
                **kwargs,
            )

        ax.set_xlabel(self.variable.x_label, plot_style.xlabel_pos)
        ax.set_ylabel(self._get_y_label(normed=normed, evts_or_cands=y_label), plot_style.ylabel_pos)

        if draw_legend:
            self.draw_legend(axis=ax, inside=legend_inside, y_axis_scale=y_axis_scale)

        return ax


class StackedHistogramPlot(HistogramPlot):
    def __init__(
        self,
        variable: HistVariable,
    ) -> None:
        super().__init__(variable=variable)

    def add_component(
        self,
        label: str,
        data: DataInputType,
        weights: WeightsInputType = None,
        color: Optional[str] = None,
        alpha: float = 1.0,
    ) -> None:
        self._add_component(
            label=label,
            histogram_key="stacked_histogram",
            data=data,
            weights=weights,
            systematics=None,
            hist_type="stepfilled",
            color=color,
            alpha=alpha,
        )

    def plot_on(
        self,
        ax: Optional[AxesType] = None,
        draw_legend: bool = True,
        legend_inside: bool = True,
        y_axis_scale: float = 1.6,
        y_label: str = "Events",
    ) -> AxesType:

        if ax is None:
            _, ax = plt.subplots()
        self._last_figure = ax.get_figure()

        histogram = list(self._histograms.histograms)[0]  # type: Histogram

        ax.hist(
            x=[histogram.binning.bin_mids[0] for _ in histogram.get_bin_counts()],
            bins=self.bin_edges,
            weights=histogram.get_bin_counts(),
            stacked=True,
            edgecolor="black",
            lw=0.3,
            color=histogram.colors,
            label=histogram.labels,  # type: ignore  # The type here is correct!
            histtype=histogram.hist_type,
        )

        # TODO: Include uncertainties if available and wanted!

        ax.set_xlabel(self._variable.x_label, plot_style.xlabel_pos)
        ax.set_ylabel(self._get_y_label(normed=False, evts_or_cands=y_label), plot_style.ylabel_pos)

        if draw_legend:
            self.draw_legend(axis=ax, inside=legend_inside, y_axis_scale=y_axis_scale)

        return ax


class DataMCHistogramPlot(HistogramPlot):
    valid_styles = ["stacked", "summed"]  # type: List[str]
    valid_ratio_types = ["normal", "vs_uncert"]  # type: List[str]

    data_key = "data_histogram"  # type: str
    mc_key = "mc_histogram"  # type: str

    legend_cols_default = 2  # type: int
    legend_loc_default = 0  # type: int

    def __init__(self, variable: HistVariable) -> None:
        super().__init__(variable=variable)

        self._special_binning = None  # type: Union[None, BinsInputType, Binning]

    def add_data_component(
        self,
        label: str,
        data: DataInputType,
        color: str = plot_style.KITColors.kit_black,
        special_binning: Union[None, BinsInputType, Binning] = None,
    ) -> None:
        if self.data_key in self._histograms.histogram_keys:
            raise RuntimeError(f"A data component has already been added to {self.__class__.__name__} instance!")
        if special_binning:
            if not self._special_binning:
                self._special_binning = special_binning
            else:
                assert self._special_binning == special_binning, (self._special_binning, special_binning)

        self._add_component(
            label=label,
            histogram_key=self.data_key,
            data=data,
            special_binning=special_binning,
            weights=None,
            systematics=None,
            hist_type="stepfilled",  # TODO: Define new own hist_type for data plots!
            color=color,
            alpha=1.0,
        )

    def add_mc_component(
        self,
        label: str,
        data: DataInputType,
        weights: WeightsInputType = None,
        systematics: SystematicsInputType = None,
        color: Optional[str] = None,
        special_binning: Union[None, BinsInputType, Binning] = None,
    ) -> None:
        if special_binning:
            if not self._special_binning:
                self._special_binning = special_binning
            else:
                assert self._special_binning == special_binning, (self._special_binning, special_binning)

        self._add_component(
            label=label,
            histogram_key=self.mc_key,
            data=data,
            special_binning=special_binning,
            weights=weights,
            systematics=systematics,
            hist_type="stepfilled",
            color=color,
            alpha=1.0,
        )

    # Same as add_mc_component method; just added to satisfy requirements from base class.
    def add_component(
        self,
        label: str,
        data: DataInputType,
        weights: WeightsInputType = None,
        systematics: SystematicsInputType = None,
        color: Optional[str] = None,
    ) -> None:
        self._add_component(
            label=label,
            histogram_key=self.mc_key,
            data=data,
            weights=weights,
            systematics=systematics,
            hist_type="stepfilled",
            color=color,
            alpha=1.0,
        )

    def plot_on(
        self,
        ax1: Optional[AxesType] = None,
        ax2: Optional[AxesType] = None,
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
        y_scale: float = 1.1,
    ) -> DataMCComparisonOutputType:
        ax1, ax2 = self._check_axes_input(ax1=ax1, ax2=ax2)
        self._last_figure = ax1.get_figure()

        self._check_style_settings_input(style=style, ratio_type=ratio_type, gof_check_method=gof_check_method)

        plot_outlier_indicators = self._check_outlier_indicator_setting(
            outlier_indicator_setting=plot_outlier_indicators,
            ratio_type=ratio_type,
        )

        if adaptive_binning:
            self._histograms.apply_adaptive_binning_based_on_key(
                key=self.mc_key,
                minimal_bin_count=5,
                minimal_number_of_bins=7,
            )

        bin_scaling = self.binning.get_bin_scaling()  # type: np.ndarray

        mc_bin_count, mc_uncert_sq, stat_mc_uncert_sq, norm_factor = self.get_bin_info_for_component(
            component_key=self.mc_key,
            data_key=self.data_key,
            normalize_to_data=normalize_to_data,
            include_sys=include_sys,
        )  # type: np.ndarray, np.ndarray, np.ndarray, float

        data_bin_count = self._histograms[self.data_key].get_bin_count_of_component(index=0)  # type: np.ndarray

        if style.lower() == "stacked":
            if len(self._histograms[self.mc_key].labels) > 1:
                stacked_component_labels = self._histograms[self.mc_key].labels  # type: Tuple[str, ...]
            else:
                stacked_component_labels = tuple(
                    [h_label + r" $\times$ " + f"{norm_factor:.2f}" for h_label in self._histograms[self.mc_key].labels]
                )

            ax1.hist(
                x=[self.bin_mids for _ in range(self._histograms[self.mc_key].number_of_components)],
                bins=self.bin_edges,
                weights=self._histograms[self.mc_key].get_bin_counts(factor=bin_scaling * norm_factor),
                stacked=True,
                edgecolor="black",
                lw=0.3,
                color=self._histograms[self.mc_key].colors,
                label=stacked_component_labels,  # type: ignore  # The type here is correct!
                histtype="stepfilled",
            )

            ax1.bar(
                x=self.bin_mids,
                height=2 * np.sqrt(mc_uncert_sq),
                width=self.bin_widths,
                bottom=mc_bin_count * bin_scaling - np.sqrt(mc_uncert_sq),
                color="black",
                hatch="///////",
                fill=False,
                lw=0,
                label="MC stat. unc." if not include_sys else "MC stat. + sys. unc.",
            )
        elif style.lower() == "summed":
            ax1.bar(
                x=self.bin_mids,
                height=2 * np.sqrt(mc_uncert_sq),
                width=self.bin_widths,
                bottom=mc_bin_count * bin_scaling - np.sqrt(mc_uncert_sq),
                color=sum_color,
                lw=0,
                label="MC" if not normalize_to_data else r"MC $\times$ " + f"{norm_factor:.2f}",
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
            label=self._histograms[self.data_key].labels[0],
        )

        try:
            comparison_output = self.do_goodness_of_fit_test(
                method=gof_check_method,
                mc_bin_count=mc_bin_count,
                data_bin_count=data_bin_count,
                stat_mc_uncertainty_sq=stat_mc_uncert_sq,
                mc_is_normalized_to_data=normalize_to_data,
            )  # type: DataMCComparisonOutputType
        except IndexError:
            logging.warning(
                f"Could not run goodness of fit check with {gof_check_method} method "
                f"for variable {self.variable.df_label}! Reverting to check with pearson method!"
            )
            comparison_output = self.do_goodness_of_fit_test(
                method="pearson",
                mc_bin_count=mc_bin_count,
                data_bin_count=data_bin_count,
                stat_mc_uncertainty_sq=stat_mc_uncert_sq,
                mc_is_normalized_to_data=normalize_to_data,
            )

        if draw_legend:
            if style == "stacked":
                self.draw_legend(
                    axis=ax1,
                    inside=legend_inside,
                    loc=legend_loc,
                    ncols=legend_cols,
                    font_size="smaller",
                    y_axis_scale=y_scale,
                )
            else:
                self.draw_legend(
                    axis=ax1,
                    inside=legend_inside,
                    loc=legend_loc,
                    ncols=legend_cols,
                    y_axis_scale=y_scale,
                )

        ax1.set_ylabel(self._get_y_label(normed=False, evts_or_cands=y_label), plot_style.ylabel_pos)
        ax2.set_xlabel(self._variable.x_label, plot_style.xlabel_pos)

        self.add_residual_ratio_plot(
            axis=ax2,
            ratio_type=ratio_type,
            data_bin_count=data_bin_count,
            mc_bin_count=mc_bin_count,
            mc_error_sq=mc_uncert_sq,
            markers_with_width=markers_with_width,
            systematics_are_included=include_sys,
            marker_color=plot_style.KITColors.kit_black,
            include_outlier_info=True,
            plot_outlier_indicators=plot_outlier_indicators,
        )

        plt.subplots_adjust(hspace=0.08)

        return comparison_output

    def get_bin_info_for_component(
        self,
        component_key: Optional[str] = None,
        data_key: Optional[str] = None,
        normalize_to_data: bool = False,
        include_sys: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        if component_key is None:
            component_key = self.mc_key
        if component_key not in self._histograms.histogram_keys:
            raise KeyError(
                f"Histogram key '{component_key}' was not added to the {self.__class__.__name__} "
                f"instance!\nAvailable histogram keys are: {self._histograms.histogram_keys}"
            )

        if data_key is None:
            data_key = self.data_key
        if data_key not in self._histograms.histogram_keys:
            raise KeyError(
                f"Histogram key '{data_key}' was not added to the {self.__class__.__name__} "
                f"instance!\nAvailable histogram keys are: {self._histograms.histogram_keys}"
            )

        component_bin_count = np.sum(np.array(self._histograms[component_key].get_bin_counts()), axis=0)

        if not normalize_to_data:
            norm_factor = 1.0  # type: float
        else:
            norm_factor = self._histograms[data_key].raw_data_size / self._histograms[component_key].raw_weight_sum
            component_bin_count *= norm_factor

        component_stat_uncert_sq = self._histograms[component_key].get_statistical_uncertainty_per_bin(
            normalization_factor=norm_factor
        )
        component_uncert_sq = copy.deepcopy(component_stat_uncert_sq)

        if include_sys:
            sys_uncertainty_squared = self._histograms[component_key].get_systematic_uncertainty_per_bin()
            if sys_uncertainty_squared is not None:
                component_uncert_sq += sys_uncertainty_squared

        assert len(component_bin_count.shape) == 1, component_bin_count.shape
        assert component_bin_count.shape[0] == self.number_of_bins, (component_bin_count.shape, self.number_of_bins)
        assert component_bin_count.shape == component_uncert_sq.shape, (
            component_bin_count.shape,
            component_uncert_sq.shape,
        )

        return component_bin_count, component_uncert_sq, component_stat_uncert_sq, norm_factor

    def do_goodness_of_fit_test(
        self,
        method: Optional[str],
        mc_bin_count: np.ndarray,
        data_bin_count: np.ndarray,
        stat_mc_uncertainty_sq: np.ndarray,
        mc_is_normalized_to_data: bool,
    ) -> DataMCComparisonOutputType:
        if method is None:
            return None

        dof = self.number_of_bins - 1 if mc_is_normalized_to_data else self.number_of_bins

        if method.lower() == "pearson":
            chi2, ndf, p_val = pearson_chi2_test(data=data_bin_count, expectation=mc_bin_count, dof=dof)
            return DataMCComparisonOutput(chi2=chi2, ndf=ndf, p_val=p_val, test_method=method, toy_output=None)
        elif method.lower() == "cowan":
            chi2, ndf, p_val = cowan_binned_likelihood_gof(data=data_bin_count, expectation=mc_bin_count, dof=dof)
            return DataMCComparisonOutput(chi2=chi2, ndf=ndf, p_val=p_val, test_method=method, toy_output=None)
        elif method.lower() == "toys":
            chi2, p_val, toy_output = toy_chi2_test(
                data=data_bin_count,
                expectation=mc_bin_count,
                error=np.where(data_bin_count >= 1, data_bin_count, np.ones(data_bin_count.shape)),
                mc_cov=self._histograms[self.mc_key].get_covariance_matrix() + np.diag(data_bin_count),
                use_text_book_approach=True,
            )
            return DataMCComparisonOutput(chi2=chi2, ndf=dof, p_val=p_val, test_method=method, toy_output=toy_output)
        elif method.lower() == "toys_inverted":
            chi2, p_val, toy_output = toy_chi2_test(
                data=data_bin_count,
                expectation=mc_bin_count,
                error=np.where(data_bin_count >= 1, data_bin_count, np.ones(data_bin_count.shape)),
                mc_cov=self._histograms[self.mc_key].get_covariance_matrix(),
            )
            return DataMCComparisonOutput(chi2=chi2, ndf=dof, p_val=p_val, test_method=method, toy_output=toy_output)
        else:
            raise ValueError(
                f"The provided goodness of fit method identifier '{method}' is not valid!\n"
                f"It must be one of {DataMCComparisonOutput.valid_test_methods()}!"
            )

    def add_residual_ratio_plot(
        self,
        axis: AxesType,
        ratio_type: str,
        data_bin_count: np.ndarray,
        mc_bin_count: np.ndarray,
        mc_error_sq: np.ndarray,
        markers_with_width: bool = True,
        systematics_are_included: bool = False,
        marker_color: str = plot_style.KITColors.kit_black,
        include_outlier_info: bool = False,
        plot_outlier_indicators: bool = False,
    ) -> None:
        if ratio_type.lower() == "normal":
            axis.set_ylabel(r"$\frac{\mathrm{Data - MC}}{\mathrm{Data}}$")
        elif ratio_type.lower() == "vs_uncert":
            if systematics_are_included:
                axis.set_ylabel(r"$\frac{\mathrm{Data - MC}}{\sigma_\mathrm{stat + sys}^\mathrm{Data - MC}}$")
            else:
                axis.set_ylabel(r"$\frac{\mathrm{Data - MC}}{\sigma_\mathrm{stat}^\mathrm{Data - MC}}$")
        else:
            raise ValueError(
                f"The provided ratio_type '{ratio_type}' is not valid!\n"
                f"The ratio_type must be one of {DataMCHistogramPlot.valid_ratio_types}!"
            )

        try:
            uh_data = unp.uarray(data_bin_count, np.sqrt(data_bin_count))
            uh_mc = unp.uarray(mc_bin_count, np.sqrt(mc_error_sq))

            if ratio_type.lower() == "normal":
                divisor = copy.deepcopy(uh_data)
            elif ratio_type.lower() == "vs_uncert":
                divisor = unp.uarray(unp.std_devs(uh_data - uh_mc), 0.0)
            else:
                divisor = None

            divisor[divisor == 0] = ufloat(0.01, 0.1)
            ratio = (uh_data - uh_mc) / divisor
            ratio[(uh_data == 0.0) & (uh_mc == 0.0)] = ufloat(0.0, 0.0)

            if ratio_type.lower() == "normal":
                ratio[np.logical_xor((uh_data == 0.0), (uh_mc == 0.0))] = ufloat(-99, 0.0)
                _max_val = 1.0  # type: Union[None, float, np.ndarray]
                assert isinstance(_max_val, float), (type(_max_val).__name__, _max_val)
                axis.set_ylim(bottom=-1.0 * _max_val, top=1.0 * _max_val)
            elif ratio_type.lower() == "vs_uncert":
                max_val_mask = (uh_data != 0.0) & (uh_mc != 0.0) & ((uh_data - uh_mc) != 0)
                if np.sum(max_val_mask) == 0:
                    _max_val = None
                    warning_str = "Max value for ratio plot cannot be determined, as no valid values are available!"
                    if include_outlier_info:
                        include_outlier_info = False
                        warning_str += " The option 'include_outlier_info' was set to False due to this!"
                    warnings.warn(warning_str)
                else:
                    _max_val = np.around(
                        max(
                            abs(
                                float(
                                    np.amin(unp.nominal_values(ratio[max_val_mask]) - unp.std_devs(ratio[max_val_mask]))
                                )
                            ),
                            abs(
                                float(
                                    np.amax(unp.nominal_values(ratio[max_val_mask]) + unp.std_devs(ratio[max_val_mask]))
                                )
                            ),
                        ),
                        decimals=1,
                    )
                    assert isinstance(_max_val, float), (type(_max_val).__name__, _max_val)
                    axis.set_ylim(bottom=-1.0 * _max_val, top=_max_val)
            else:
                _max_val = None

            axis.axhline(y=0, color=plot_style.KITColors.dark_grey, alpha=0.8)
            axis.errorbar(
                self.bin_mids,
                unp.nominal_values(ratio),
                yerr=unp.std_devs(ratio),
                xerr=self.bin_widths / 2 if markers_with_width else None,
                ls="",
                marker=".",
                color=marker_color,
            )

            if not include_outlier_info:
                return

            assert isinstance(_max_val, float), (type(_max_val).__name__, _max_val)
            max_val = _max_val  # type: float

            for bin_mid, r_val, mc_val, data_val in zip(self.bin_mids, ratio, uh_mc, uh_data):
                if mc_val == 0.0 and (
                    (data_val != 0.0 and ratio_type.lower() != "vs_uncert")
                    or (abs(r_val) > max_val and ratio_type.lower() == "vs_uncert")
                ):
                    axis.text(x=bin_mid, y=+0.1 * max_val, s="No MC", fontsize=5, rotation=90, ha="center", va="bottom")
                    axis.text(
                        x=bin_mid,
                        y=+0.1 * max_val,
                        s=f"#Data={int(unp.nominal_values(data_val))}",
                        fontsize=5,
                        rotation=90,
                        ha="center",
                        va="bottom",
                    )
                elif data_val == 0.0 and (
                    (mc_val != 0.0 and ratio_type.lower() != "vs_uncert")
                    or (abs(r_val) > max_val and ratio_type.lower() == "vs_uncert")
                ):
                    axis.text(
                        x=bin_mid,
                        y=+0.1 * max_val,
                        s=f"#MC={unp.nominal_values(mc_val):.0f}",
                        fontsize=5,
                        rotation=90,
                        ha="center",
                        va="bottom",
                    )
                    axis.text(x=bin_mid, y=-0.1 * max_val, s="No Data", fontsize=5, rotation=90, ha="center", va="top")
                elif r_val > 1.0 and plot_outlier_indicators:
                    axis.text(
                        x=bin_mid,
                        y=+0.08 * max_val,
                        s=f"{unp.nominal_values(r_val):3.2f}" + r"$\rightarrow$",
                        fontsize=5,
                        rotation=90,
                        ha="right",
                        va="bottom",
                    )
                elif r_val < -1.0 and plot_outlier_indicators:
                    axis.text(
                        x=bin_mid,
                        y=-0.08 * max_val,
                        s=r"$\leftarrow$" + f"{unp.nominal_values(r_val):3.2f}",
                        fontsize=5,
                        rotation=90,
                        ha="right",
                        va="top",
                    )
                else:
                    pass

        except ZeroDivisionError:
            axis.text(
                x=self.bin_mids[int(np.ceil(len(self.bin_mids) / 2.0))],
                y=0.1,
                s="DataMCHistogramPlot: ZeroDivisionError occurred!",
                fontsize=8,
                ha="center",
                va="bottom",
            )
            axis.axhline(y=0, color=plot_style.KITColors.dark_grey, alpha=0.8)

    @staticmethod
    def _check_style_settings_input(
        style: str,
        ratio_type: str,
        gof_check_method: Optional[str],
    ) -> None:
        if not style.lower() in DataMCHistogramPlot.valid_styles:
            raise ValueError(
                f"The argument 'style' must be one of {DataMCHistogramPlot.valid_styles}!" f"Provided was '{style}'"
            )
        if not ratio_type.lower() in DataMCHistogramPlot.valid_ratio_types:
            raise ValueError(
                f"The argument 'ratio_type' must be one of {DataMCHistogramPlot.valid_ratio_types}!"
                f"Provided was '{ratio_type}'"
            )
        if gof_check_method is not None and gof_check_method.lower() not in DataMCComparisonOutput.valid_test_methods():
            raise ValueError(
                f"The argument 'gof_check_method' must be one of {DataMCComparisonOutput.valid_test_methods()} "
                f"or None! Provided was '{gof_check_method}'"
            )

    @staticmethod
    def _check_outlier_indicator_setting(
        outlier_indicator_setting: bool,
        ratio_type: str,
    ) -> bool:
        if ratio_type.lower() == "vs_uncert":
            former_outlier_setting = outlier_indicator_setting
            outlier_indicator_setting = False
            if former_outlier_setting != outlier_indicator_setting:
                logging.info(
                    f"Resetting 'plot_outlier_indicators' from True to False, "
                    f"because of 'ratio_type' being set to '{ratio_type.lower()}'."
                )
        return outlier_indicator_setting

    @staticmethod
    def _check_axes_input(
        ax1: AxesType,
        ax2: AxesType,
    ) -> Tuple[AxesType, AxesType]:
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
        height_ratio: Tuple[float, float] = (3.5, 1),
    ) -> Tuple[figure.Figure, Tuple[AxesType, AxesType]]:
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
            sharex="all",
            gridspec_kw={"height_ratios": [height_ratio[0], height_ratio[1]]},
        )

        assert isinstance(fig, figure.Figure), type(fig)
        assert len(axs) == 2, (len(axs), axs)

        return fig, axs
