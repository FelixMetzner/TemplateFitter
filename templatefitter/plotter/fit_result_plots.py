"""
Plotting tools to illustrate fit results produced with this package
"""
import copy
import logging
import itertools
import numpy as np

from collections import defaultdict
from matplotlib import pyplot as plt
from typing import Optional, Union, Tuple, List, Dict, Any

from templatefitter.binned_distributions.binning import Binning
from templatefitter.fit_model.channel import Channel, DataChannelContainer
from templatefitter.binned_distributions.binned_distribution import DataColumnNamesInput

from templatefitter.plotter import plot_style
from templatefitter.plotter.histogram_variable import HistVariable
from templatefitter.plotter.histogram_plot_base import HistogramPlot, AxesType

from templatefitter.fit_model.model_builder import FitModel

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "FitResultPlot",
    "FitResultPlotter"
]

plot_style.set_matplotlibrc_params()

# TODO: Option to add Chi2 test
# TODO: Option to add ratio plot

# TODO: Provide option to use dict to add latex labels to channels with mapping {channel.name: latex_label_raw_string}
# TODO: Provide option to use dict for colors of with mapping {template.name: color_string}

# TODO: Maybe implement method to generate BinnedDistribution from slice or projection of original BinnedDistribution.


class FitResultPlot(HistogramPlot):
    valid_styles = ["stacked", "summed"]
    valid_ratio_types = ["normal", "vs_uncert"]
    valid_gof_methods = ["pearson", "cowan", "toys"]

    data_key = "data_histogram"
    mc_key = "mc_histogram"
    valid_histogram_keys = [data_key, mc_key]
    required_histogram_keys = valid_histogram_keys

    def __init__(self, variable: HistVariable, binning: Binning) -> None:
        super().__init__(variable=variable)

        self._binning = binning

    @property
    def binning(self) -> Optional[Binning]:
        return self._binning

    def add_component(
            self,
            label: str,
            histogram_key: str,
            bin_counts: np.ndarray,
            bin_errors_squared: np.ndarray,
            data_column_names: DataColumnNamesInput,
            color: str
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
            alpha=1.0
        )

    def plot_on(
            self,
            ax1: Optional[AxesType] = None,
            style: str = "stacked",
            include_sys: bool = False,
            markers_with_width: bool = True,
            sum_color: str = plot_style.KITColors.kit_purple,
            draw_legend: bool = True,
            legend_inside: bool = True,
            legend_cols: Optional[int] = None,
            legend_loc: Optional[Union[int, str]] = None,
            y_scale: float = 1.1
    ) -> Any:
        self._check_required_histograms()

        bin_scaling = 1. / np.around(self.bin_widths / self.minimal_bin_width, decimals=0)

        data_bin_count = self._histograms[self.data_key].get_bin_count_of_component(index=0)
        data_bin_errors_sq = self._histograms[self.data_key].get_histogram_squared_bin_errors_of_component(index=0)

        mc_bin_counts = self._histograms[self.mc_key].get_bin_counts()
        mc_sum_bin_count = np.sum(np.array(mc_bin_counts), axis=0)
        mc_sum_bin_error_sq = self._histograms[self.mc_key].get_statistical_uncertainty_per_bin()

        if style.lower() == "stacked":
            ax1.hist(
                x=[self.bin_mids for _ in range(self._histograms[self.mc_key].number_of_components)],
                bins=self.bin_edges,
                weights=mc_bin_counts,
                stacked=True,
                edgecolor="black",
                lw=0.3,
                color=self._histograms[self.mc_key].colors,
                label=self._histograms[self.mc_key].labels,
                histtype='stepfilled'
            )

            ax1.bar(
                x=self.bin_mids,
                height=2 * np.sqrt(mc_sum_bin_error_sq),
                width=self.bin_widths,
                bottom=mc_sum_bin_count * bin_scaling - np.sqrt(mc_sum_bin_error_sq),
                color="black",
                hatch="///////",
                fill=False,
                lw=0,
                label="MC stat. unc." if not include_sys else "MC stat. + sys. unc."
            )
        elif style.lower() == "summed":
            ax1.bar(
                x=self.bin_mids,
                height=2 * np.sqrt(mc_sum_bin_error_sq),
                width=self.bin_widths,
                bottom=mc_sum_bin_count * bin_scaling - np.sqrt(mc_sum_bin_error_sq),
                color=sum_color,
                lw=0,
                label="MC stat. unc." if not include_sys else "MC stat. + sys. unc."
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
            label=self._histograms[self.data_key].labels[0]
        )

        if draw_legend:
            if style == "stacked":
                self.draw_legend(axis=ax1, inside=legend_inside, loc=legend_loc, ncols=legend_cols,
                                 font_size="smaller", y_axis_scale=y_scale)
            else:
                self.draw_legend(axis=ax1, inside=legend_inside, loc=legend_loc, ncols=legend_cols,
                                 y_axis_scale=y_scale)

        ax1.set_ylabel(self._get_y_label(normed=False), plot_style.ylabel_pos)
        ax1.set_xlabel(self._variable.x_label, plot_style.xlabel_pos)

    def _check_histogram_key(self, histogram_key: str) -> None:
        assert isinstance(histogram_key, str), type(histogram_key)
        if histogram_key not in self.valid_histogram_keys:
            raise RuntimeError(f"Invalid histogram_key provided!\n"
                               f"The histogram key must be one of {self.valid_histogram_keys}!\n"
                               f"However, you provided the histogram_key {histogram_key}!")

    def _check_required_histograms(self) -> None:
        for required_hist_key in self.required_histogram_keys:
            if required_hist_key not in self._histograms.histogram_keys:
                raise RuntimeError(f"The required histogram key '{required_hist_key}' is not available!\n"
                                   f"Available histogram keys: {list(self._histograms.keys())}\n"
                                   f"Required histogram keys: {self.required_histogram_keys}")


class FitResultPlotter:

    def __init__(
            self,
            variable: HistVariable,
            fit_model: FitModel,
            reference_dimension: int = 0,
            fig_size: Tuple[float, float] = (5, 5),  # TODO: could be handled via kwargs
            involved_hist_variables: Optional[List[HistVariable]] = None,
            **kwargs
    ) -> None:
        self._variable = variable
        self._fit_model = fit_model
        self._reference_dimension = reference_dimension
        # TODO: Reference dimension and respective variable of the fit model must be same as self.variable!

        self._fig_size = fig_size
        self._optional_arguments_dict = kwargs  # type: Dict[str, Any]

        self._channel_name_list = []  # type: List[str]
        self._channel_variables = {}  # type: Dict[str, HistVariable]
        self._channel_sub_bin_mapping = defaultdict(list)  # type: Dict[str: List[Optional[Tuple[int, ...]]]]

        self._involved_hist_variables = involved_hist_variables  # type: Optional[List[HistVariable]]

        self._get_histograms_from_model(fit_model=fit_model)

    def plot_fit_result(self, use_initial_values: bool = False) -> None:
        for mc_channel in self._fit_model.mc_channels_to_plot:
            current_binning = mc_channel.binning.get_binning_for_one_dimension(dimension=self.reference_dimension)
            data_column_name_for_plot = mc_channel.data_column_names[self.reference_dimension]

            data_channel = self._fit_model.data_channels_to_plot.get_channel_by_name(name=mc_channel.name)
            data_bin_count = data_channel.bin_counts
            data_bin_errors_squared = data_channel.bin_errors_sq

            # TODO: Use channel_latex_labels for plot info...

            for sub_bin_ids in self._channel_sub_bin_mapping[mc_channel.name]:
                if sub_bin_ids is None:
                    sub_bin_info = None
                else:
                    sub_bin_info = self._get_bin_infos(channel_name=mc_channel.name, bin_ids=sub_bin_ids)
                nd_array_slices = self._get_slices(bins_in_other_dims=sub_bin_ids)

                current_plot = FitResultPlot(variable=self.channel_variables[mc_channel.name], binning=current_binning)

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
                        color=self._get_mc_color(key=template.process_name, original_color=template.color)
                    )

                subset_data_bin_count = data_bin_count[nd_array_slices]
                subset_data_bin_errors_squared = data_bin_errors_squared[nd_array_slices]

                current_plot.add_component(
                    label=self._get_data_label(),
                    histogram_key=FitResultPlot.data_key,
                    bin_counts=subset_data_bin_count,
                    bin_errors_squared=subset_data_bin_errors_squared,
                    data_column_names=data_column_name_for_plot,
                    color=self._get_data_color()
                )

                fig, axs = plt.subplots(nrows=1, ncols=1, figsize=self._fig_size, dpi=200)
                current_plot.plot_on(
                    ax1=axs,
                    #  style=???,  # str = "stacked",
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
                if sub_bin_info:
                    axs.set_title(sub_bin_info, loc="left", fontsize=8, color=plot_style.KITColors.dark_grey)

    def _get_histograms_from_model(self, fit_model: FitModel) -> None:

        for mc_channel in fit_model.mc_channels_to_plot:
            self._channel_name_list.append(mc_channel.name)

            ch_binning = mc_channel.binning.get_binning_for_one_dimension(dimension=self.reference_dimension)
            self._add_channel_hist_var(channel_name=mc_channel.name, original_binning=ch_binning)

            ch_data_column_names = mc_channel.data_column_names
            data_column_name_for_plot = ch_data_column_names[self.reference_dimension]
            assert data_column_name_for_plot == self.variable.df_label, \
                (data_column_name_for_plot, self.variable.df_label, ch_data_column_names)

            full_binning = mc_channel.binning

            other_binnings_by_dimension = {
                dim: full_binning.num_bins[dim]
                for dim in range(full_binning.dimensions) if dim != self.reference_dimension
            }
            assert list(set(other_binnings_by_dimension.keys())) == list(other_binnings_by_dimension.keys()), \
                other_binnings_by_dimension

            if len(other_binnings_by_dimension) == 0:
                self._channel_sub_bin_mapping[mc_channel.name].append(None)
            else:
                bin_ranges = [list(range(n_bins)) for n_bins in other_binnings_by_dimension.values()]
                all_combinations = list(itertools.product(*bin_ranges))
                self._channel_sub_bin_mapping[mc_channel.name].extend(all_combinations)

        assert list(set(self._channel_name_list)) == self._channel_name_list, self._channel_name_list

        channel_label_check_list = copy.copy(self._channel_name_list)
        for data_channel in fit_model.data_channels_to_plot:
            data_channel_base_name = DataChannelContainer.get_base_channel_name(data_channel_name=data_channel.name)
            if data_channel_base_name not in channel_label_check_list:
                raise RuntimeError(f"Encountered channel in data channels of fit model, which is not known:\n"
                                   f"\tData Channel name: {data_channel_base_name}\n"
                                   f"List of known channels:\n\t-" + "\n\t-".join(self._channel_name_list))
            else:
                channel_label_check_list.remove(data_channel_base_name)
            assert data_channel_base_name in self._channel_sub_bin_mapping.keys(), \
                (data_channel_base_name, self._channel_sub_bin_mapping.keys())

            ch_binning = data_channel.binning.get_binning_for_one_dimension(dimension=self.reference_dimension)
            self._compare_binning_to_channel_variable_binning(channel_name=data_channel_base_name, binning=ch_binning)

            ch_data_column_names = data_channel.data_column_names
            data_column_name_for_plot = ch_data_column_names[self.reference_dimension]
            assert data_column_name_for_plot == self.variable.df_label, \
                (data_column_name_for_plot, self.variable.df_label, ch_data_column_names)

    @property
    def variable(self) -> HistVariable:
        return self._variable

    @property
    def channel_names(self) -> List[str]:
        return self._channel_name_list

    @property
    def channel_variables(self) -> Dict[str, HistVariable]:
        return self._channel_variables

    @property
    def number_of_channels(self) -> int:
        return len(self._channel_name_list)

    @property
    def reference_dimension(self) -> int:
        return self._reference_dimension

    def _add_channel_hist_var(self, channel_name: str, original_binning: Binning) -> None:
        assert channel_name not in self._channel_variables.keys(), (channel_name, self._channel_variables.keys())

        assert original_binning.dimensions == 1, original_binning.dimensions

        assert len(original_binning.num_bins) == 1, original_binning.num_bins
        assert original_binning.num_bins[0] == original_binning.num_bins_total, \
            (original_binning.num_bins, original_binning.num_bins_total)

        assert len(original_binning.range) == 1, original_binning.range

        assert len(original_binning.log_scale_mask) == 1, original_binning.log_scale_mask

        channel_hist_var = HistVariable(
            df_label=self.variable.df_label,
            n_bins=original_binning.num_bins_total,
            scope=original_binning.range[0],
            var_name=self.variable.variable_name,
            unit=self.variable.unit,
            use_log_scale=original_binning.log_scale_mask[0]
        )

        self._channel_variables[channel_name] = channel_hist_var

    def _get_channel_label(self, channel: Channel) -> Optional[str]:
        if "channel_label_dict" in self._optional_arguments_dict:
            channel_label_dict = self._optional_arguments_dict["channel_label_dict"]
            assert isinstance(channel_label_dict, dict), (channel_label_dict, type(channel_label_dict))
            assert all(isinstance(k, str) for k in channel_label_dict.keys()), list(channel_label_dict.keys())
            assert all(isinstance(v, str) for v in channel_label_dict.values()), list(channel_label_dict.values())
            if channel.name not in channel_label_dict.keys():
                raise KeyError(f"No entry for the channel {channel.name} in the provided channel_label_dict!\n"
                               f"channel_label_dict:\n\t"
                               + "\n\t".join([f"{k}: {v}" for k, v in channel_label_dict.items()]))
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

    def _get_mc_color(self, key: str, original_color: Optional[str]) -> Optional[str]:
        if "mc_color_dict" in self._optional_arguments_dict:
            mc_color_dict = self._optional_arguments_dict["mc_color_dict"]
            assert isinstance(mc_color_dict, dict), (mc_color_dict, type(mc_color_dict))
            assert all(isinstance(k, str) for k in mc_color_dict.keys()), list(mc_color_dict.keys())
            assert all(isinstance(v, str) for v in mc_color_dict.values()), list(mc_color_dict.values())
            if key not in mc_color_dict.keys():
                raise KeyError(f"No entry for the key {key} in the provided mc_color_dict!\n"
                               f"mc_color_dict:\n\t" + "\n\t".join([f"{k}: {v}" for k, v in mc_color_dict.items()]))
            return mc_color_dict[key]
        else:
            return original_color

    def _get_mc_label(self, key: str, original_label: Optional[str]) -> Optional[str]:
        if "mc_label_dict" in self._optional_arguments_dict:
            mc_label_dict = self._optional_arguments_dict["mc_label_dict"]
            assert isinstance(mc_label_dict, dict), (mc_label_dict, type(mc_label_dict))
            assert all(isinstance(k, str) for k in mc_label_dict.keys()), list(mc_label_dict.keys())
            assert all(isinstance(v, str) for v in mc_label_dict.values()), list(mc_label_dict.values())
            if key not in mc_label_dict.keys():
                raise KeyError(f"No entry for the key {key} in the provided mc_label_dict!\n"
                               f"mc_label_dict:\n\t" + "\n\t".join([f"{k}: {v}" for k, v in mc_label_dict.items()]))
            return mc_label_dict[key]
        else:
            return original_label

    def _compare_binning_to_channel_variable_binning(self, channel_name: str, binning: Binning) -> None:
        assert channel_name in self.channel_variables, (channel_name, list(self.channel_variables.keys()))
        ch_variable = self.channel_variables[channel_name]

        assert binning.dimensions == 1, binning.dimensions

        assert len(binning.num_bins) == 1, binning.num_bins
        assert binning.num_bins[0] == binning.num_bins_total, (binning.num_bins, binning.num_bins_total)

        assert binning.num_bins_total == ch_variable.n_bins, (binning.num_bins_total, ch_variable.n_bins)

        assert len(binning.range) == 1, binning.range
        assert binning.range[0] == ch_variable.scope, (binning.range, ch_variable.scope)

        assert len(binning.log_scale_mask) == 1, binning.log_scale_mask
        assert binning.log_scale_mask[0] == ch_variable.use_log_scale, \
            (binning.log_scale_mask, ch_variable.use_log_scale)

    @staticmethod
    def _get_histogram_name(channel_name: str, is_data: bool = False, bin_id: Union[str, int, None] = None) -> str:
        if is_data:
            name = f"channel_{channel_name}_data"
        else:
            name = f"channel_{channel_name}_mc"

        if bin_id is not None:
            name += f"_bin_{bin_id}"

        return name

    def _get_slices(self, bins_in_other_dims: Optional[Tuple[int, ...]]) -> Tuple[slice, ...]:
        if bins_in_other_dims is None:
            assert self.reference_dimension == 0
            return tuple([slice(None)])

        slice_list = []
        for dim in range(len(bins_in_other_dims) + 1):
            if dim == self.reference_dimension:
                slice_list.append(slice(None))
            else:
                slice_list.append(bins_in_other_dims[dim if dim < self.reference_dimension else dim - 1])

        assert len(slice_list) == len(bins_in_other_dims) + 1, (len(slice_list), len(bins_in_other_dims))
        return tuple(slice_list)

    def _get_involved_hist_variable_latex_label(self, column_name: str) -> Optional[str]:
        if self._involved_hist_variables is None:
            return None
        names = [h_var.variable_name for h_var in self._involved_hist_variables if h_var.df_label == column_name]
        assert len(names) <= 1, (len(names), names)
        if len(names) == 1:
            return names[0]
        return None

    def _get_bin_infos(self, channel_name: str, bin_ids: Tuple[int, ...]) -> str:
        dimensions = [i if i < self.reference_dimension else i + 1 for i in range(len(bin_ids))]
        assert len(bin_ids) == len(dimensions), (len(bin_ids), len(dimensions))

        mc_channel = self._fit_model.mc_channels_to_plot.get_channel_by_name(name=channel_name)
        data_column_names = [mc_channel.data_column_names[dim] for dim in dimensions]
        latex_names = [self._get_involved_hist_variable_latex_label(column_name=name) for name in data_column_names]

        string_list = []
        for dim, l_name, bin_n in zip(dimensions, latex_names, bin_ids):
            if l_name is not None:
                string_list.append(rf"{l_name} (dim. {dim + 1}): Bin {bin_n+ 1}")
            else:
                string_list.append(f"Dimension {dim + 1}: Bin {bin_n + 1}")

        return "\n".join(string_list)
