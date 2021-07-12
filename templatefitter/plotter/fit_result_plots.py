"""
Plotting tools to illustrate fit results produced with this package
"""
import os
import logging
import numpy as np

from matplotlib import pyplot as plt
from typing import Optional, Union, Tuple, List, Dict, Type

from templatefitter.utility import PathType
from templatefitter.binned_distributions.binning import Binning
from templatefitter.binned_distributions.binned_distribution import DataColumnNamesInput

from templatefitter.plotter import plot_style
from templatefitter.plotter.plot_utilities import export, AxesType
from templatefitter.plotter.histogram_variable import HistVariable
from templatefitter.plotter.fit_plots_base import FitPlotBase, FitPlotterBase

from templatefitter.fit_model.model_builder import FitModel

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "FitResultPlot",
    "FitResultPlotter",
]

plot_style.set_matplotlibrc_params()


# TODO: Option to add Chi2 test
# TODO: Option to add ratio plot


class FitResultPlot(FitPlotBase):
    valid_styles = ["stacked", "summed"]  # type: List[str]
    valid_ratio_types = ["normal", "vs_uncert"]  # type: List[str]
    valid_gof_methods = ["pearson", "cowan", "toys"]  # type: List[str]

    data_key = "data_histogram"  # type: str
    mc_key = "mc_histogram"  # type: str
    valid_histogram_keys = [data_key, mc_key]  # type: List[str]
    required_histogram_keys = valid_histogram_keys  # type: List[str]

    def __init__(
        self,
        variable: HistVariable,
        binning: Binning,
    ) -> None:
        super().__init__(
            variable=variable,
            binning=binning,
        )

    def add_component(
        self,
        label: str,
        histogram_key: str,
        bin_counts: np.ndarray,
        bin_errors_squared: np.ndarray,
        data_column_names: DataColumnNamesInput,
        color: str,
    ) -> None:
        self._check_histogram_key(histogram_key=histogram_key)

        self._add_prebinned_component(
            label=label,
            histogram_key=histogram_key,
            bin_counts=bin_counts,
            original_binning=self._binning,
            bin_errors_squared=bin_errors_squared,
            data_column_names=data_column_names,
            hist_type="stepfilled",  # TODO: Define own hist_type for data plots: histogram_key == data_key!
            color=color,
            alpha=1.0,
        )

    def plot_on(
        self,
        ax1: AxesType,
        style: str = "stacked",
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

        data_bin_count = self._histograms[self.data_key].get_bin_count_of_component(index=0)
        data_bin_errors_sq = self._histograms[self.data_key].get_histogram_squared_bin_errors_of_component(index=0)

        mc_bin_counts = self._histograms[self.mc_key].get_bin_counts(factor=bin_scaling)
        # clean_mc_bin_counts = [np.where(bc < 0., 0., bc) for bc in mc_bin_counts]

        mc_sum_bin_count = np.sum(np.array(mc_bin_counts), axis=0)
        mc_sum_bin_error_sq = self._histograms[self.mc_key].get_statistical_uncertainty_per_bin()

        bar_bottom = mc_sum_bin_count - np.sqrt(mc_sum_bin_error_sq)
        height_corr = np.where(bar_bottom < 0.0, bar_bottom, 0.0)
        bar_bottom[bar_bottom < 0.0] = 0.0
        bar_height = 2 * np.sqrt(mc_sum_bin_error_sq) - height_corr

        if style.lower() == "stacked":
            ax1.hist(
                x=[self.bin_mids for _ in range(self._histograms[self.mc_key].number_of_components)],
                bins=self.bin_edges,
                weights=mc_bin_counts,
                stacked=True,
                edgecolor="black",
                lw=0.3,
                color=self._histograms[self.mc_key].colors,
                label=self._histograms[self.mc_key].labels,  # type: ignore  # The type here is correct!
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
        elif style.lower() == "summed":
            ax1.bar(
                x=self.bin_mids,
                height=bar_height,
                width=self.bin_widths,
                bottom=bar_bottom,
                color=sum_color,
                lw=0,
                label="MC stat. unc." if not include_sys else "MC stat. + sys. unc.",
            )
        else:
            raise RuntimeError(f"Invalid style '{style.lower()}!'\n style must be one of {self.valid_styles}!")

        ax1.errorbar(
            x=self.bin_mids,
            y=data_bin_count * bin_scaling,
            yerr=np.sqrt(data_bin_errors_sq),
            xerr=self.bin_widths / 2 if markers_with_width else None,
            ls="",
            marker=".",
            color="black",
            label=self._histograms[self.data_key].labels[0],
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

        ax1.set_ylabel(self._get_y_label(normed=False), plot_style.ylabel_pos)
        ax1.set_xlabel(self._variable.x_label, plot_style.xlabel_pos)

    def _check_histogram_key(
        self,
        histogram_key: str,
    ) -> None:
        assert isinstance(histogram_key, str), type(histogram_key)
        if histogram_key not in self.valid_histogram_keys:
            raise RuntimeError(
                f"Invalid histogram_key provided!\n"
                f"The histogram key must be one of {self.valid_histogram_keys}!\n"
                f"However, you provided the histogram_key {histogram_key}!"
            )

    def _check_required_histograms(self) -> None:
        for required_hist_key in self.required_histogram_keys:
            if required_hist_key not in self._histograms.histogram_keys:
                raise RuntimeError(
                    f"The required histogram key '{required_hist_key}' is not available!\n"
                    f"Available histogram keys: {list(self._histograms.histogram_keys)}\n"
                    f"Required histogram keys: {self.required_histogram_keys}"
                )


class FitResultPlotter(FitPlotterBase):
    def __init__(
        self,
        variables_by_channel: Union[Dict[str, Tuple[HistVariable, ...]], Tuple[HistVariable, ...]],
        fit_model: FitModel,
        reference_dimension: int = 0,
        fig_size: Tuple[float, float] = (5, 5),
        **kwargs,
    ) -> None:
        super().__init__(
            variables_by_channel=variables_by_channel,
            fit_model=fit_model,
            reference_dimension=reference_dimension,
            fig_size=fig_size,
            **kwargs,
        )
        self._plotter_class = FitResultPlot  # type: Type[FitPlotBase]
        self.mc_key = FitResultPlot.mc_key  # type: str
        self.data_key = FitResultPlot.data_key  # type: str

    def plot_fit_result(
        self,
        use_initial_values: bool = False,
        reference_dimension: Optional[int] = None,
        output_dir_path: Optional[PathType] = None,
        output_name_tag: Optional[str] = None,
        bin_info_location: Optional[str] = None,
    ) -> Dict[str, List[PathType]]:
        output_lists = {
            "pdf": [],
            "png": [],
        }  # type: Dict[str, List[PathType]]

        if (output_dir_path is None) != (output_name_tag is None):
            raise ValueError(
                "Parameter 'output_name_tag' and 'output_dir_path' must either both be provided or both set to None!"
            )

        bin_info_pos = "right" if bin_info_location is None else bin_info_location  # type: str
        ref_dim = self.base_reference_dimension if reference_dimension is None else reference_dimension  # type: int

        for mc_channel in self._fit_model.mc_channels_to_plot:
            current_binning = mc_channel.binning.get_binning_for_one_dimension(dimension=ref_dim)
            data_column_name_for_plot = mc_channel.data_column_names[ref_dim]

            data_channel = self._fit_model.data_channels_to_plot.get_channel_by_name(name=mc_channel.name)

            assert data_channel.bin_counts is not None
            data_bin_count = data_channel.bin_counts  # type: np.ndarray
            assert data_channel.bin_errors_sq is not None
            data_bin_errors_squared = data_channel.bin_errors_sq  # type: np.ndarray

            for counter, sub_bin_info in enumerate(
                self._get_sub_bin_infos_for(
                    channel_name=mc_channel.name,
                    reference_dimension=ref_dim,
                )
            ):
                sub_bin_info_text = self._get_sub_bin_info_text(
                    channel_name=mc_channel.name,
                    sub_bin_infos=sub_bin_info,
                    reference_dimension=ref_dim,
                )

                nd_array_slices = self._get_slices(reference_dimension=ref_dim, sub_bin_info=sub_bin_info)

                current_plot = self.plotter_class(
                    variable=self.channel_variables(dimension=ref_dim)[mc_channel.name],
                    binning=current_binning,
                )

                for template in mc_channel.templates:
                    template_bin_count = template.expected_bin_counts(use_initial_values=use_initial_values)
                    template_bin_error_sq = template.expected_bin_errors_squared(use_initial_values=use_initial_values)

                    subset_bin_count = template_bin_count[nd_array_slices]
                    subset_bin_errors_squared = template_bin_error_sq[nd_array_slices]

                    current_plot.add_component(
                        label=self._get_mc_label(key=template.process_name, original_label=template.latex_label),
                        histogram_key=self.mc_key,
                        bin_counts=subset_bin_count,
                        bin_errors_squared=subset_bin_errors_squared,
                        data_column_names=data_column_name_for_plot,
                        color=self._get_mc_color(key=template.process_name, original_color=template.color),
                    )

                subset_data_bin_count = data_bin_count[nd_array_slices]
                subset_data_bin_errors_squared = data_bin_errors_squared[nd_array_slices]

                current_plot.add_component(
                    label=self._get_data_label(),
                    histogram_key=self.data_key,
                    bin_counts=subset_data_bin_count,
                    bin_errors_squared=subset_data_bin_errors_squared,
                    data_column_names=data_column_name_for_plot,
                    color=self._get_data_color(),
                )

                fig, axs = plt.subplots(nrows=1, ncols=1, figsize=self._fig_size, dpi=200)
                current_plot.plot_on(
                    ax1=axs,
                    #  style=???,  # str = "stacked",  # TODO: Include summed style
                    #  include_sys=???,  # bool = False,
                    #  markers_with_width=???,  # bool = True,
                    #  sum_color=???,  # str = plot_style.KITColors.kit_purple,
                    #  draw_legend=???,  # bool = True,
                    #  legend_inside=???,  # bool = True,
                    #  legend_cols=???,  # Optional[int] = None,
                    #  legend_loc=???,  # Optional[Union[int, str]] = None,
                    #  y_scale=???,  # float = 1.1
                )

                if bin_info_pos == "left" or sub_bin_info_text is None:
                    axs.set_title(self._get_channel_label(channel=mc_channel), loc="right")
                else:
                    fig.suptitle(self._get_channel_label(channel=mc_channel), x=0.97, horizontalalignment="right")

                if sub_bin_info_text is not None:
                    info_title = sub_bin_info_text
                    if axs.get_ylim()[1] > 0.85e4 and bin_info_pos == "left":
                        padding = " " * 9
                        info_title = "\n".join([padding + info for info in sub_bin_info_text.split("\n")])

                    if bin_info_pos == "right":
                        info_title = r"$\;$" + "\n" + r"$\;$" + "\n" + info_title

                    axs.set_title(info_title, loc=bin_info_pos, fontsize=6, color=plot_style.KITColors.dark_grey)

                if output_dir_path is not None:
                    assert output_name_tag is not None

                    add_info = ""
                    if use_initial_values:
                        add_info = "_with_initial_values"
                    filename = f"fit_result_plot_{output_name_tag}_{mc_channel.name}_bin_{counter}{add_info}"

                    export(fig=fig, filename=filename, target_dir=output_dir_path, close_figure=True)
                    output_lists["pdf"].append(os.path.join(output_dir_path, f"{filename}.pdf"))
                    output_lists["png"].append(os.path.join(output_dir_path, f"{filename}.png"))

        return output_lists

    def plot_fit_result_projections(
        self,
        project_to: int,
        use_initial_values: bool = False,
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
            binning = mc_channel.binning.get_binning_for_one_dimension(dimension=project_to)
            data_column_name_for_plot = mc_channel.data_column_names[project_to]

            data_channel = self._fit_model.data_channels_to_plot.get_channel_by_name(name=mc_channel.name)

            data_bin_count, data_bin_errors_squared = data_channel.project_onto_dimension(
                bin_counts=data_channel.bin_counts,
                dimension=project_to,
                bin_errors_squared=data_channel.bin_errors_sq,
            )

            plot = self.plotter_class(
                variable=self.channel_variables(dimension=project_to)[mc_channel.name],
                binning=binning,
            )

            for template in mc_channel.templates:
                template_bin_count, template_bin_error_sq = template.project_onto_dimension(
                    bin_counts=template.expected_bin_counts(use_initial_values=use_initial_values),
                    dimension=project_to,
                    bin_errors_squared=template.expected_bin_errors_squared(use_initial_values=use_initial_values),
                )

                plot.add_component(
                    label=self._get_mc_label(key=template.process_name, original_label=template.latex_label),
                    histogram_key=self.mc_key,
                    bin_counts=template_bin_count,
                    bin_errors_squared=template_bin_error_sq,
                    data_column_names=data_column_name_for_plot,
                    color=self._get_mc_color(key=template.process_name, original_color=template.color),
                )

            plot.add_component(
                label=self._get_data_label(),
                histogram_key=self.data_key,
                bin_counts=data_bin_count,
                bin_errors_squared=data_bin_errors_squared,
                data_column_names=data_column_name_for_plot,
                color=self._get_data_color(),
            )

            fig, axs = plt.subplots(nrows=1, ncols=1, figsize=self._fig_size, dpi=200)
            plot.plot_on(
                ax1=axs,
                #  style=???,  # str = "stacked",  # TODO: Include summed style
                #  include_sys=???,  # bool = False,
                #  markers_with_width=???,  # bool = True,
                #  sum_color=???,  # str = plot_style.KITColors.kit_purple,
                #  draw_legend=???,  # bool = True,
                #  legend_inside=???,  # bool = True,
                #  legend_cols=???,  # Optional[int] = None,
                #  legend_loc=???,  # Optional[Union[int, str]] = None,
                #  y_scale=???,  # float = 1.1
            )

            axs.set_title(self._get_channel_label(channel=mc_channel), loc="right")

            if output_dir_path is not None:
                assert output_name_tag is not None

                add_info = ""
                if use_initial_values:
                    add_info = "_with_initial_values"
                filename = f"fit_result_plot_{output_name_tag}_{mc_channel.name}_dim_{project_to}_projection{add_info}"

                export(fig=fig, filename=filename, target_dir=output_dir_path, close_figure=True)
                output_lists["pdf"].append(os.path.join(output_dir_path, f"{filename}.pdf"))
                output_lists["png"].append(os.path.join(output_dir_path, f"{filename}.png"))

        return output_lists
