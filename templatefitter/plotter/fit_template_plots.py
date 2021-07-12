"""
Plotting tools to visualize fit templates
"""

import os
import logging
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import colors as mpl_colors
from matplotlib import ticker as mpl_ticker
from itertools import combinations as iter_combinations
from typing import Union, Optional, Tuple, List, Dict, Type

from templatefitter.utility import PathType

from templatefitter.binned_distributions.binning import Binning
from templatefitter.binned_distributions.binned_distribution import DataColumnNamesInput

from templatefitter.fit_model.channel import Channel
from templatefitter.fit_model.template import Template

from templatefitter.plotter import plot_style
from templatefitter.plotter.plot_utilities import export, AxesType
from templatefitter.plotter.histogram_variable import HistVariable
from templatefitter.plotter.fit_plots_base import FitPlotBase, FitPlotterBase

from templatefitter.fit_model.model_builder import FitModel

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "FitTemplatePlot",
    "FitTemplatesPlotter",
]


plot_style.set_matplotlibrc_params()


class FitTemplatePlot(FitPlotBase):
    required_hist_key = "template"
    hist_key = required_hist_key

    def __init__(
        self,
        variable: HistVariable,
        binning: Binning,
    ) -> None:
        super().__init__(
            variable=variable,
            binning=binning,
        )

        self._has_component = False  # type: bool

    def add_component(
        self,
        label: str,
        bin_counts: np.ndarray,
        bin_errors_squared: np.ndarray,
        data_column_names: DataColumnNamesInput,
        color: str,
    ) -> None:
        if self._has_component:
            raise RuntimeError("A component was already added, and a FitTemplatePlot only has one component...")

        self._add_prebinned_component(
            label=label,
            histogram_key=self.required_hist_key,
            bin_counts=bin_counts,
            original_binning=self._binning,
            bin_errors_squared=bin_errors_squared,
            data_column_names=data_column_names,
            hist_type="stepfilled",
            color=color,
            alpha=1.0,
        )
        self._has_component = True

    def plot_on(
        self,
        ax1: AxesType,
        include_sys: bool = False,
        markers_with_width: bool = True,
        sum_color: str = plot_style.KITColors.kit_purple,
        draw_legend: bool = True,
        legend_inside: bool = True,
        legend_cols: Optional[int] = None,
        legend_loc: Optional[Union[int, str]] = None,
        y_scale: float = 1.1,
    ) -> None:
        self._check_required_histograms()

        bin_scaling = self.binning.get_bin_scaling()  # type: np.ndarray

        template_bin_counts = self._histograms[self.hist_key].get_bin_counts(factor=bin_scaling)
        assert isinstance(template_bin_counts, list) and len(template_bin_counts) == 1, template_bin_counts

        mc_sum_bin_error_sq = self._histograms[self.hist_key].get_statistical_uncertainty_per_bin()

        bar_bottom = template_bin_counts[0] - np.sqrt(mc_sum_bin_error_sq)
        height_corr = np.where(bar_bottom < 0.0, bar_bottom, 0.0)
        bar_bottom[bar_bottom < 0.0] = 0.0
        bar_height = 2 * np.sqrt(mc_sum_bin_error_sq) - height_corr

        ax1.hist(
            x=[self.bin_mids for _ in range(self._histograms[self.hist_key].number_of_components)],
            bins=self.bin_edges,
            weights=template_bin_counts,
            stacked=True,
            edgecolor="black",
            lw=0.3,
            color=self._histograms[self.hist_key].colors,
            label=self._histograms[self.hist_key].labels,  # type: ignore  # The type here is correct!
            histtype="stepfilled",
        )

        ax1.bar(
            x=self.bin_mids,
            height=bar_height,
            width=self.bin_widths,
            bottom=bar_bottom,
            color="black",
            hatch="///////",
            fill=False,
            lw=0,
            label="MC stat. unc." if not include_sys else "MC stat. + sys. unc.",
        )

        if draw_legend:
            self.draw_legend(
                axis=ax1,
                inside=legend_inside,
                loc=legend_loc,
                ncols=legend_cols,
                font_size="smaller",
                y_axis_scale=y_scale,
            )

        ax1.set_ylabel(self._get_y_label(normed=False), plot_style.ylabel_pos)
        ax1.set_xlabel(self._variable.x_label, plot_style.xlabel_pos)

    def _check_required_histograms(self) -> None:
        if self.required_hist_key not in self._histograms.histogram_keys:
            raise RuntimeError(
                f"The required histogram key '{self.required_hist_key}' is not available!\n"
                f"Available histogram keys: {list(self._histograms.histogram_keys)}\n"
                f"Required histogram keys: {self.required_hist_key}"
            )


class FitTemplatesPlotter(FitPlotterBase):
    plot_name_prefix = "fit_template_plot"  # type: str
    plot_2d_name_prefix = "fit_template_2d_plot"  # type: str

    default_2d_c_map_base_color = plot_style.KITColors.light_grey  # type: str

    def __init__(
        self,
        variables_by_channel: Union[Dict[str, Tuple[HistVariable, ...]], Tuple[HistVariable, ...]],
        fit_model: FitModel,
        fig_size: Tuple[float, float] = (5, 5),
        **kwargs,
    ) -> None:
        super().__init__(
            variables_by_channel=variables_by_channel,
            fit_model=fit_model,
            reference_dimension=0,
            fig_size=fig_size,
            **kwargs,
        )

        self._plotter_class = FitTemplatePlot  # type: Type[FitPlotBase]

    def plot_projected_templates(
        self,
        use_initial_values: bool = True,
        output_dir_path: Optional[PathType] = None,
        output_name_tag: Optional[str] = None,
    ) -> Dict[str, List[PathType]]:
        output_lists = {
            "pdf": [],
            "png": [],
        }  # type: Dict[str, List[PathType]]

        if (output_dir_path is None) != (output_name_tag is None):
            raise ValueError(
                "Parameter 'output_name_tag' and 'output_dir_path' must either both be provided or both set to None!"
            )

        for mc_channel in self._fit_model.mc_channels_to_plot:
            for dimension in range(mc_channel.binning.dimensions):
                current_binning = mc_channel.binning.get_binning_for_one_dimension(dimension=dimension)
                data_column_name_for_plot = mc_channel.data_column_names[dimension]

                for template in mc_channel.templates:
                    current_plot = self.plotter_class(
                        variable=self.channel_variables(dimension=dimension)[mc_channel.name],
                        binning=current_binning,
                    )

                    template_bin_count, template_bin_error_sq = template.project_onto_dimension(
                        bin_counts=template.expected_bin_counts(use_initial_values=use_initial_values),
                        dimension=dimension,
                        bin_errors_squared=template.expected_bin_errors_squared(use_initial_values=use_initial_values),
                    )

                    current_plot.add_component(
                        label=self._get_template_label(template=template),
                        bin_counts=template_bin_count,
                        bin_errors_squared=template_bin_error_sq,
                        data_column_names=data_column_name_for_plot,
                        color=self._get_template_color(key=template.process_name, original_color=template.color),
                    )

                    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=self._fig_size, dpi=200)
                    current_plot.plot_on(ax1=ax)

                    ax.set_title(self._get_plot_title(template=template, channel=mc_channel), loc="right")

                    if output_dir_path is not None:
                        assert output_name_tag is not None

                        add_info = ""
                        if use_initial_values:
                            add_info = "_with_initial_values"

                        template_info = f"{mc_channel.name}_template_{template.process_name}{add_info}"
                        filename = f"{self.plot_name_prefix}_{output_name_tag}_dim_{dimension}_{template_info}"

                        export(fig=fig, filename=filename, target_dir=output_dir_path, close_figure=True)
                        output_lists["pdf"].append(os.path.join(output_dir_path, f"{filename}.pdf"))
                        output_lists["png"].append(os.path.join(output_dir_path, f"{filename}.png"))

        return output_lists

    def plot_2d_templates(
        self,
        use_initial_values: bool = True,
        output_dir_path: Optional[PathType] = None,
        output_name_tag: Optional[str] = None,
        base_color: Union[None, str, Dict[str, str]] = None,
        alternative_temp_color: Optional[Dict[str, str]] = None,
    ) -> Dict[str, List[PathType]]:
        output_lists = {
            "pdf": [],
            "png": [],
        }  # type: Dict[str, List[PathType]]

        if (output_dir_path is None) != (output_name_tag is None):
            raise ValueError(
                "Parameter 'output_name_tag' and 'output_dir_path' must either both be provided or both set to None!"
            )

        if isinstance(base_color, dict):
            c_map_base_color = base_color  # type: Union[str, Dict[str, str]]
        else:
            c_map_base_color = self.default_2d_c_map_base_color if base_color is None else base_color
        alt_temp_color_dict = {} if alternative_temp_color is None else alternative_temp_color  # type: Dict[str, str]

        for mc_channel in self._fit_model.mc_channels_to_plot:
            for dim_pair in iter_combinations(list(range(mc_channel.binning.dimensions)), 2):
                current_binning = mc_channel.binning.get_binning_for_x_dimensions(dimensions=dim_pair)

                x_variable = self.channel_variables(dimension=dim_pair[0])[mc_channel.name]  # type: HistVariable
                y_variable = self.channel_variables(dimension=dim_pair[1])[mc_channel.name]  # type: HistVariable

                for template in mc_channel.templates:
                    template_bin_count, template_bin_error_sq = template.project_onto_two_dimensions(
                        bin_counts=template.expected_bin_counts(use_initial_values=use_initial_values),
                        dimensions=dim_pair,
                        bin_errors_squared=template.expected_bin_errors_squared(use_initial_values=use_initial_values),
                    )

                    template_color = self._get_template_color(key=template.process_name, original_color=template.color)

                    bin_scaling_tuple = current_binning.get_bin_scaling_per_dim_tuple()  # type: Tuple[np.ndarray, ...]
                    bin_scaling = np.outer(*bin_scaling_tuple)  # type: np.ndarray
                    assert len(bin_scaling.shape) == 2, bin_scaling.shape
                    assert bin_scaling.shape[0] == bin_scaling_tuple[0].shape[0], (
                        bin_scaling.shape,
                        bin_scaling_tuple[0].shape,
                    )
                    assert bin_scaling.shape[1] == bin_scaling_tuple[1].shape[0], (
                        bin_scaling.shape,
                        bin_scaling_tuple[1].shape,
                    )

                    assert template_bin_count.shape == bin_scaling.shape, (template_bin_count.shape, bin_scaling.shape)
                    value_matrix = template_bin_count * bin_scaling  # type: np.ndarray

                    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=self._fig_size, dpi=200)

                    c_values = [0.0, np.max(value_matrix)]  # type: List[float]

                    if np.sum(template_bin_count) == 0:
                        c_values = [0.0, 1.0]
                        value_matrix = np.ones_like(value_matrix) * 0.8

                    c_map_temp_color = alt_temp_color_dict.get(template_color, template_color)
                    if isinstance(c_map_base_color, dict):
                        base_c = c_map_base_color.get(template_color, self.default_2d_c_map_base_color)  # type: str
                        colors = [base_c, c_map_temp_color]  # type: List[str]
                    else:
                        colors = [c_map_base_color, c_map_temp_color]

                    c_norm = plt.Normalize(min(c_values), max(c_values))
                    c_tuples = list(zip(map(c_norm, c_values), colors))
                    color_map = mpl_colors.LinearSegmentedColormap.from_list(name="", colors=c_tuples)

                    color_map.set_bad(color=plot_style.KITColors.white)
                    value_matrix[value_matrix == 0.0] = np.nan
                    heatmap = ax.imshow(X=np.flip(value_matrix.T, axis=0), cmap=color_map, aspect="auto")
                    plt.colorbar(heatmap)

                    self._set_2d_axis_tick_labels(ax=ax, binning=current_binning)

                    ax.set_title(self._get_plot_title(template=template, channel=mc_channel), loc="right")

                    ax.set_xlabel(x_variable.x_label, plot_style.xlabel_pos)
                    ax.set_ylabel(y_variable.x_label, plot_style.ylabel_pos)

                    if np.sum(template_bin_count) == 0:
                        ax.text(
                            x=0.5,
                            y=0.5,
                            s="No Data",
                            fontsize="x-large",
                            ha="center",
                            va="center",
                            zorder=10,
                            transform=ax.transAxes,
                        )

                    if output_dir_path is not None:
                        assert output_name_tag is not None

                        add_info = ""
                        if use_initial_values:
                            add_info = "_with_initial_values"

                        dims_str = f"{dim_pair[0]}_{dim_pair[1]}"  # type: str
                        template_info = f"{mc_channel.name}_template_{template.process_name}{add_info}"
                        filename = f"{self.plot_2d_name_prefix}_{output_name_tag}_dim_{dims_str}_{template_info}"

                        export(fig=fig, filename=filename, target_dir=output_dir_path, close_figure=True)
                        output_lists["pdf"].append(os.path.join(output_dir_path, f"{filename}.pdf"))
                        output_lists["png"].append(os.path.join(output_dir_path, f"{filename}.png"))

        return output_lists

    @staticmethod
    def _set_2d_axis_tick_labels(ax: AxesType, binning: Binning) -> None:
        x_tick_positions = np.arange(binning.num_bins[0] + 1) - 0.5  # type: np.array
        y_tick_positions = np.arange(binning.num_bins[1] + 1) - 0.5  # type: np.array
        x_tick_labels = np.array(binning.bin_edges[0])  # type: np.array
        y_tick_labels = np.array(binning.bin_edges[1])  # type: np.array

        ax.set_xticks(ticks=x_tick_positions)
        ax.set_yticks(ticks=y_tick_positions)

        ax.xaxis.set_minor_locator(locator=mpl_ticker.NullLocator())
        ax.yaxis.set_minor_locator(locator=mpl_ticker.NullLocator())

        x_labels = [f"{la:.2f}" for la in x_tick_labels]  # type: List[str]
        y_labels = [f"{la:.2f}" for la in y_tick_labels[::-1]]  # type: List[str]
        ax.set_xticklabels(labels=x_labels, rotation=-45, ha="left")
        ax.set_yticklabels(labels=y_labels)

    def _get_plot_title(
        self,
        template: Template,
        channel: Channel,
    ) -> str:
        template_label = self._get_template_label(template=template)

        if self._fit_model.number_of_channels > 1:
            channel_label = self._get_channel_label(channel=channel)
            return f"{template_label} in Channel {channel_label}"
        else:
            return template_label

    def _get_template_color(
        self,
        key: str,
        original_color: str,
    ) -> str:
        return self._get_attribute_from_optional_arguments_dict(
            attribute_name="template_color_dict", key=key, default_value=original_color
        )

    def _get_template_label(
        self,
        template: Template,
    ) -> str:
        return self._get_attribute_from_optional_arguments_dict(
            attribute_name="mc_label_dict", key=template.process_name, default_value=template.latex_label
        )
