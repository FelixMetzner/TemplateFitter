"""
Plotting tools to visualize fit templates
"""

import os
import logging
import numpy as np

from matplotlib import pyplot as plt
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
    # "FitTemplates2DHeatMapPlot",  # TODO
    # "FitTemplates2DHeatMapPlotter",  # TODO
]

# TODO: Allow for irregular binning! -> Fix issue with _add_prebinned_component

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

        bin_scaling = 1.0 / np.around(self.bin_widths / self.minimal_bin_width, decimals=0)

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

    def __init__(
        self,
        variables: Tuple[HistVariable, ...],
        fit_model: FitModel,
        fig_size: Tuple[float, float] = (5, 5),
        **kwargs,
    ) -> None:
        super().__init__(
            variables=variables,
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


# class FitTemplates2DHeatMapPlot(FitPlotBase):
#     # TODO Generalize heat map plot in validation plots!
#     pass
#
#
# class FitTemplates2DHeatMapPlotter(FitPlotterBase):
#     pass
