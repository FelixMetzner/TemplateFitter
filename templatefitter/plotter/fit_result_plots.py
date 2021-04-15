"""
Plotting tools to illustrate fit results produced with this package
"""
import os
import copy
import logging
import itertools
import numpy as np

from matplotlib import pyplot as plt
from typing import Optional, Union, Tuple, List, Dict, NamedTuple, Generator, Any

from templatefitter.utility import PathType
from templatefitter.fit_model.channel import Channel
from templatefitter.binned_distributions.binning import Binning
from templatefitter.fit_model.data_channel import DataChannelContainer
from templatefitter.binned_distributions.binned_distribution import DataColumnNamesInput

from templatefitter.plotter import plot_style
from templatefitter.plotter.plot_utilities import export, AxesType
from templatefitter.plotter.histogram_variable import HistVariable
from templatefitter.plotter.histogram_plot_base import HistogramPlot

from templatefitter.fit_model.model_builder import FitModel

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "FitResultPlot",
    "FitResultPlotter",
]

plot_style.set_matplotlibrc_params()


# TODO: Option to add Chi2 test
# TODO: Option to add ratio plot


class FitResultPlot(HistogramPlot):
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
        super().__init__(variable=variable)

        self._binning = binning  # type: Binning

    @property
    def binning(self) -> Binning:
        return self._binning

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

        bin_scaling = 1.0 / np.around(self.bin_widths / self.minimal_bin_width, decimals=0)

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


class SubBinInfos(NamedTuple):
    bin_ids: Tuple[int, ...]
    bin_edges: Tuple[Tuple[float, float], ...]


class FitResultPlotter:
    def __init__(
        self,
        variables: Tuple[HistVariable, ...],
        fit_model: FitModel,
        reference_dimension: int = 0,
        fig_size: Tuple[float, float] = (5, 5),  # TODO: could be handled via kwargs
        **kwargs,
    ) -> None:
        self._variables = variables  # type: Tuple[HistVariable, ...]

        self._fit_model = fit_model  # type: FitModel

        self._reference_dimension = reference_dimension  # type: int

        self._fig_size = fig_size  # type: Tuple[float, float]
        self._optional_arguments_dict = kwargs  # type: Dict[str, Any]

        self._channel_name_list = []  # type: List[str]
        self._channel_variables_per_dim = {}  # type: Dict[str, Dict[int, HistVariable]]
        self._number_of_bins_per_ch_per_dim = {}  # type: Dict[str, Dict[int, int]]

        self._get_histograms_from_model(fit_model=fit_model)

    def plot_fit_result(
        self,
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
            # TODO: Maybe also use the inverse of the reference dimension!?
            current_binning = mc_channel.binning.get_binning_for_one_dimension(dimension=self.reference_dimension)
            data_column_name_for_plot = mc_channel.data_column_names[self.reference_dimension]

            data_channel = self._fit_model.data_channels_to_plot.get_channel_by_name(name=mc_channel.name)

            assert data_channel.bin_counts is not None
            data_bin_count = data_channel.bin_counts  # type: np.ndarray
            assert data_channel.bin_errors_sq is not None
            data_bin_errors_squared = data_channel.bin_errors_sq  # type: np.ndarray

            for counter, sub_bin_info in enumerate(
                self._get_sub_bin_infos_for(
                    channel_name=mc_channel.name,
                    reference_dimension=self.reference_dimension,  # TODO
                )
            ):
                sub_bin_info_text = self._get_sub_bin_info_text(
                    channel_name=mc_channel.name,
                    sub_bin_infos=sub_bin_info,
                    reference_dimension=self.reference_dimension,
                )

                nd_array_slices = self._get_slices(sub_bin_info=sub_bin_info)

                current_plot = FitResultPlot(
                    variable=self.channel_variables(dimension=self.reference_dimension)[mc_channel.name],
                    binning=current_binning,
                )

                for template in mc_channel.templates:
                    template_bin_count = template.expected_bin_counts(use_initial_values=use_initial_values)
                    template_bin_error_sq = template.expected_bin_errors_squared(use_initial_values=use_initial_values)

                    subset_bin_count = template_bin_count[nd_array_slices]
                    subset_bin_errors_squared = template_bin_error_sq[nd_array_slices]

                    current_plot.add_component(
                        label=self._get_mc_label(key=template.process_name, original_label=template.latex_label),
                        histogram_key=FitResultPlot.mc_key,
                        bin_counts=subset_bin_count,
                        bin_errors_squared=subset_bin_errors_squared,
                        data_column_names=data_column_name_for_plot,
                        color=self._get_mc_color(key=template.process_name, original_color=template.color),
                    )

                subset_data_bin_count = data_bin_count[nd_array_slices]
                subset_data_bin_errors_squared = data_bin_errors_squared[nd_array_slices]

                current_plot.add_component(
                    label=self._get_data_label(),
                    histogram_key=FitResultPlot.data_key,
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

                bin_info_pos = "right"  # TODO: Make this variable accessible

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

            plot = FitResultPlot(
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
                    histogram_key=FitResultPlot.mc_key,
                    bin_counts=template_bin_count,
                    bin_errors_squared=template_bin_error_sq,
                    data_column_names=data_column_name_for_plot,
                    color=self._get_mc_color(key=template.process_name, original_color=template.color),
                )

            plot.add_component(
                label=self._get_data_label(),
                histogram_key=FitResultPlot.data_key,
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

    def _get_histograms_from_model(
        self,
        fit_model: FitModel,
    ) -> None:
        for mc_channel in fit_model.mc_channels_to_plot:
            ch_column_names = mc_channel.data_column_names
            assert len(ch_column_names) == len(self.variables), (len(ch_column_names), len(self.variables))
            assert all(c == v.df_label for c, v in zip(ch_column_names, self.variables)), [
                (c, v.df_label) for c, v in zip(ch_column_names, self.variables)
            ]

            self._channel_name_list.append(mc_channel.name)
            self._add_channel_hist_vars(channel_name=mc_channel.name, original_binning=mc_channel.binning)

            bins_per_dim = {dim: n_bins for dim, n_bins in enumerate(mc_channel.binning.num_bins)}
            self._number_of_bins_per_ch_per_dim.update({mc_channel.name: bins_per_dim})

        assert len(set(self._channel_name_list)) == len(self._channel_name_list), self._channel_name_list
        assert len(self._channel_variables_per_dim) == len(self._channel_name_list), (
            self._channel_name_list,
            list(self._channel_variables_per_dim.keys()),
        )
        channel_label_check_list = copy.copy(self._channel_name_list)

        for data_channel in fit_model.data_channels_to_plot:
            data_channel_base_name = DataChannelContainer.get_base_channel_name(data_channel_name=data_channel.name)
            if data_channel_base_name not in channel_label_check_list:
                raise RuntimeError(
                    f"Encountered channel in data channels of fit model, which is not known:\n"
                    f"\tData Channel name: {data_channel_base_name}\n"
                    f"List of known channels:\n\t-" + "\n\t-".join(self._channel_name_list)
                )
            else:
                channel_label_check_list.remove(data_channel_base_name)

            self._compare_binning_to_channel_variable_binning(
                channel_name=data_channel_base_name,
                binning=data_channel.binning,
            )

            for dim, column_name in enumerate(data_channel.data_column_names):
                var = self._channel_variables_per_dim[data_channel_base_name][dim]
                assert column_name == var.df_label, (column_name, var.df_label, dim, data_channel.name)

        assert all(
            len(self._number_of_bins_per_ch_per_dim[ch_name]) == len(vars_per_dim)
            for ch_name, vars_per_dim in self._channel_variables_per_dim.items()
        )

    def channel_variables(
        self,
        dimension: int,
    ) -> Dict[str, HistVariable]:
        return {ch_name: v[dimension] for ch_name, v in self._channel_variables_per_dim.items()}

    def variable(
        self,
        dimension: int,
    ) -> HistVariable:
        return self._variables[dimension]

    @property
    def variables(self) -> Tuple[HistVariable, ...]:
        return self._variables

    @property
    def channel_names(self) -> List[str]:
        return self._channel_name_list

    @property
    def channel_variables_per_dim_dict(self) -> Dict[str, Dict[int, HistVariable]]:
        return self._channel_variables_per_dim

    @property
    def number_of_channels(self) -> int:
        return len(self._channel_name_list)

    @property
    def reference_dimension(self) -> int:
        return self._reference_dimension

    def _add_channel_hist_vars(
        self,
        channel_name: str,
        original_binning: Binning,
    ) -> None:
        assert channel_name not in self._channel_variables_per_dim.keys(), (
            channel_name,
            self._channel_variables_per_dim.keys(),
        )

        channel_dim_dict = {}  # type: Dict[int, HistVariable]
        for dimension, hist_variable in enumerate(self.variables):
            binning = original_binning.get_binning_for_one_dimension(dimension=dimension)
            assert binning.dimensions == 1, binning.dimensions
            assert len(binning.num_bins) == 1, binning.num_bins
            assert binning.num_bins[0] == binning.num_bins_total, (binning.num_bins, binning.num_bins_total)
            assert len(binning.range) == 1, binning.range
            assert len(binning.log_scale_mask) == 1, binning.log_scale_mask

            channel_hist_var_for_dim = HistVariable(
                df_label=hist_variable.df_label,
                n_bins=binning.num_bins_total,
                scope=binning.range[0],
                var_name=hist_variable.variable_name,
                unit=hist_variable.unit,
                use_log_scale=binning.log_scale_mask[0],
            )
            channel_dim_dict.update({dimension: channel_hist_var_for_dim})

        self._channel_variables_per_dim[channel_name] = channel_dim_dict

    def _get_channel_label(
        self,
        channel: Channel,
    ) -> Optional[str]:
        if "channel_label_dict" in self._optional_arguments_dict:
            channel_label_dict = self._optional_arguments_dict["channel_label_dict"]
            assert isinstance(channel_label_dict, dict), (channel_label_dict, type(channel_label_dict))
            assert all(isinstance(k, str) for k in channel_label_dict.keys()), list(channel_label_dict.keys())
            assert all(isinstance(v, str) for v in channel_label_dict.values()), list(channel_label_dict.values())
            if channel.name not in channel_label_dict.keys():
                raise KeyError(
                    f"No entry for the channel {channel.name} in the provided channel_label_dict!\n"
                    f"channel_label_dict:\n\t" + "\n\t".join([f"{k}: {v}" for k, v in channel_label_dict.items()])
                )
            return channel_label_dict[channel.name]
        elif channel.latex_label is not None:
            return channel.latex_label
        else:
            return channel.name

    def _get_data_color(self) -> str:
        if "data_color" in self._optional_arguments_dict:
            data_color = self._optional_arguments_dict["data_color"]
            assert isinstance(data_color, str), (data_color, type(data_color))
            return data_color
        else:
            return plot_style.KITColors.kit_black

    def _get_data_label(self) -> str:
        if "data_label" in self._optional_arguments_dict:
            data_label = self._optional_arguments_dict["data_label"]
            assert isinstance(data_label, str), (data_label, type(data_label))
            return data_label
        else:
            return "Data"

    def _get_mc_color(
        self,
        key: str,
        original_color: str,
    ) -> str:
        return self._get_attribute_from_optional_arguments_dict(
            attribute_name="mc_color_dict",
            key=key,
            default_value=original_color,
        )

    def _get_mc_label(
        self,
        key: str,
        original_label: str,
    ) -> str:
        return self._get_attribute_from_optional_arguments_dict(
            attribute_name="mc_label_dict",
            key=key,
            default_value=original_label,
        )

    def _get_attribute_from_optional_arguments_dict(
        self,
        attribute_name: str,
        key: str,
        default_value: str,
    ) -> str:
        if attribute_name in self._optional_arguments_dict:
            attribute_dict = self._optional_arguments_dict[attribute_name]
            assert isinstance(attribute_dict, dict), (attribute_dict, type(attribute_dict))
            assert all(isinstance(k, str) for k in attribute_dict.keys()), list(attribute_dict.keys())
            assert all(isinstance(v, str) for v in attribute_dict.values()), list(attribute_dict.values())
            if key not in attribute_dict.keys():
                raise KeyError(
                    f"No entry for the key {key} in the provided attribute dictionary  {attribute_name}!\n"
                    f"{attribute_name} dictionary:\n\t" + "\n\t".join([f"{k}: {v}" for k, v in attribute_dict.items()])
                )
            return attribute_dict[key]
        else:
            return default_value

    def _compare_binning_to_channel_variable_binning(
        self,
        channel_name: str,
        binning: Binning,
    ) -> None:
        assert channel_name in self._channel_variables_per_dim.keys(), (
            channel_name,
            self._channel_variables_per_dim.keys(),
        )

        for dimension, variable in self.channel_variables_per_dim_dict[channel_name].items():
            binning_for_dim = binning.get_binning_for_one_dimension(dimension=dimension)
            assert binning_for_dim.dimensions == 1, (binning_for_dim.dimensions, variable.df_label)

            assert len(binning_for_dim.num_bins) == 1, (binning_for_dim.num_bins, variable.df_label)
            assert binning_for_dim.num_bins[0] == binning_for_dim.num_bins_total, (
                binning_for_dim.num_bins,
                binning_for_dim.num_bins_total,
                variable.df_label,
            )

            assert binning_for_dim.num_bins_total == variable.n_bins, (
                binning.num_bins_total,
                variable.n_bins,
                variable.df_label,
            )

            assert len(binning_for_dim.range) == 1, (binning_for_dim.range, variable.df_label)
            assert binning_for_dim.range[0] == variable.scope, (binning.range, variable.scope, variable.df_label)

            assert len(binning_for_dim.log_scale_mask) == 1, (binning_for_dim.log_scale_mask, variable.df_label)
            assert binning_for_dim.log_scale_mask[0] == variable.use_log_scale, (
                binning_for_dim.log_scale_mask,
                variable.use_log_scale,
                variable.df_label,
            )

    @staticmethod
    def _get_bin_edge_pairs(
        binning: Binning,
    ) -> List[Tuple[float, float]]:
        assert binning.dimensions == 1, binning.dimensions
        assert binning.num_bins[0] == binning.num_bins_total, (binning.num_bins, binning.num_bins_total)
        assert isinstance(binning.bin_edges, tuple), (type(binning.bin_edges), binning.bin_edges)
        assert len(binning.bin_edges) == 1, binning.bin_edges
        assert len(binning.bin_edges[0]) == binning.num_bins_total + 1, (
            len(binning.bin_edges[0]),
            binning.num_bins_total,
        )
        return [(binning.bin_edges[0][i], binning.bin_edges[0][i + 1]) for i in range(binning.num_bins_total)]

    def _get_sub_bin_infos_for(
        self,
        channel_name: str,
        reference_dimension: int,
    ) -> Generator[Optional[SubBinInfos], None, None]:
        for mc_channel in self._fit_model.mc_channels_to_plot:
            if mc_channel.name != channel_name:
                continue

            if mc_channel.binning.dimensions == 1:
                yield None
            else:
                bin_numbers_per_other_dim = []  # type: List[List[int]]
                bin_edges_per_other_dim = []  # type: List[List[Tuple[float, float]]]

                for dim in range(mc_channel.binning.dimensions):
                    if dim != reference_dimension:
                        bin_numbers_per_other_dim.append(list(range(mc_channel.binning.num_bins[dim])))
                        edges = self._get_bin_edge_pairs(binning=mc_channel.binning.get_binning_for_one_dimension(dim))
                        bin_edges_per_other_dim.append(edges)

                for bin_combination, _bin_edges in zip(
                    itertools.product(*bin_numbers_per_other_dim), itertools.product(*bin_edges_per_other_dim)
                ):
                    assert isinstance(bin_combination, tuple), type(bin_combination)
                    assert len(bin_combination) == mc_channel.binning.dimensions - 1, (
                        len(bin_combination),
                        mc_channel.binning.dimensions,
                    )
                    assert isinstance(_bin_edges, tuple), type(_bin_edges)
                    assert len(_bin_edges) == len(bin_combination), (len(_bin_edges), len(bin_combination))
                    assert all(isinstance(e, tuple) and len(e) == 2 for e in _bin_edges)

                    yield SubBinInfos(bin_ids=bin_combination, bin_edges=_bin_edges)

    @staticmethod
    def _get_histogram_name(
        channel_name: str,
        is_data: bool = False,
        bin_id: Union[str, int, None] = None,
    ) -> str:
        if is_data:
            name = f"channel_{channel_name}_data"  # type: str
        else:
            name = f"channel_{channel_name}_mc"

        if bin_id is not None:
            name += f"_bin_{bin_id}"

        return name

    def _get_slices(
        self,
        sub_bin_info: Optional[SubBinInfos],
    ) -> Tuple[Union[slice, int], ...]:
        if sub_bin_info is None:
            assert self.reference_dimension == 0
            return tuple([slice(None)])

        assert isinstance(sub_bin_info, SubBinInfos), type(sub_bin_info)
        bins_in_other_dims = sub_bin_info.bin_ids
        assert isinstance(bins_in_other_dims, tuple), type(bins_in_other_dims)
        assert all(isinstance(n_bins, int) for n_bins in bins_in_other_dims), bins_in_other_dims

        slice_list = []  # type: List[Union[slice, int]]
        for dim in range(len(bins_in_other_dims) + 1):
            if dim == self.reference_dimension:
                slice_list.append(slice(None))
            else:
                slice_list.append(bins_in_other_dims[dim if dim < self.reference_dimension else dim - 1])

        assert len(slice_list) == len(bins_in_other_dims) + 1, (len(slice_list), len(bins_in_other_dims))
        return tuple(slice_list)

    def _get_sub_bin_info_text(
        self,
        channel_name: str,
        sub_bin_infos: Optional[SubBinInfos],
        reference_dimension: int,
    ) -> Optional[str]:
        if sub_bin_infos is None:
            return None

        channel_vars_per_dim = self._channel_variables_per_dim[channel_name]
        dimensions = [i for i in range(len(channel_vars_per_dim)) if i != reference_dimension]

        if len(dimensions) == 0:
            return None

        assert len(sub_bin_infos.bin_ids) == len(dimensions), (len(sub_bin_infos.bin_ids), len(dimensions))

        string_list = []  # type: List[str]
        for dim, bin_id, bin_edges in zip(dimensions, sub_bin_infos.bin_ids, sub_bin_infos.bin_edges):
            n_bins_total_in_this_dim = self._number_of_bins_per_ch_per_dim[channel_name][dim]
            variable = channel_vars_per_dim[dim]

            if variable.df_label is not None:
                info_str = rf"{variable.variable_name} (Dim. {dim + 1}): Bin {bin_id + 1}/{n_bins_total_in_this_dim}"
            else:
                info_str = f"Dimension {dim + 1}: Bin {bin_id + 1}/{n_bins_total_in_this_dim}"

            info_str += f": [{bin_edges[0]:.2f}, {bin_edges[1]:.2f}]"
            if variable.unit is not None:
                info_str += rf" {variable.unit}"

            string_list.append(info_str)

        return "\n".join(string_list)
