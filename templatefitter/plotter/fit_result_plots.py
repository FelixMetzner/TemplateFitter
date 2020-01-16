"""
Plotting tools to illustrate fit results produced with this package
"""
import copy
import logging
import numpy as np
from matplotlib import pyplot as plt, figure
from typing import Optional, Union, Tuple, List, Dict, Any

from templatefitter.binned_distributions.binning import Binning
from templatefitter.binned_distributions.binned_distribution import DataInputType

from templatefitter.plotter import plot_style
from templatefitter.plotter.histogram_variable import HistVariable
from templatefitter.plotter.histogram_plot_base import HistogramPlot, AxesType

from templatefitter.fit_model.model_builder import FitModel

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "FitResultPlot"
]

plot_style.set_matplotlibrc_params()

OneOrTwoAxesType = Union[AxesType, Tuple[AxesType, AxesType]]


# TODO: Option to add Chi2 test
# TODO: Option to add ratio plot

# TODO: Option to use initial values for plotting

# TODO: Provide option to use dict to add latex labels to channels with mapping {channel.name: latex_label_raw_string}
# TODO: Provide option to use dict for colors of with mapping {template.name: color_string}


#########
# TODO: We need a HistComponent which is generated from a binned distribution. For this HistComponent, everything
#       is already fixed: Binning, BinCount, etc...
#       - we can check if the variables df_label is in the BinnedDistributions data_column_names
#       - we can check if the binning is the same as defined in the variable!
#       -> Need to differentiate between HistComponent types!
#       -> Need to update Histogram.add_histogram_component method accordingly
#
# TODO: Maybe implement method to generate BinnedDistribution from slice or projection of original BinnedDistribution.
#########

class FitResultPlot(HistogramPlot):

    def __init__(
            self,
            variable: HistVariable,
            fit_model: FitModel,
            reference_dimension: int = 0,
            **kwargs
    ) -> None:
        super().__init__(variable=variable)
        self._reference_dimension = reference_dimension
        self._optional_arguments_dict = kwargs  # type: Dict[str, Any]

        self._channel_name_list = []  # type: List[str]
        self._channel_latex_labels = {}  # type: Dict[str, str]
        self._get_histograms_from_model(fit_model=fit_model)

        self._is_initialized = True

    def _get_histograms_from_model(self, fit_model: FitModel) -> None:
        #####
        # TODO: Either the projection option has to be used, or we have to create separate Histograms per bin of all
        #       other dimensions. In any way: the dimension which should be used as reference has to be specified!
        # TODO: The bin edges and so forth have to be chosen correctly from the Binning of the FitModel to
        #       create the Histograms / the underlying BinnedDistribution
        #####

        for mc_channel in fit_model.mc_channels_to_plot:
            self._channel_name_list.append(mc_channel.name)
            channel_latex_label = self._get_channel_label(key=mc_channel.name, original_label=mc_channel.latex_label)
            self._channel_latex_labels.update({mc_channel.name: channel_latex_label})
            mc_histogram_key = f"channel_{mc_channel.name}_mc"

            ch_binning = mc_channel.binning.get_binning_for_one_dimension(dimension=self.reference_dimension)
            self._compare_original_binning_to_variable_binning(original_binning=ch_binning)

            ch_data_column_names = mc_channel.data_column_names
            data_column_name_for_plot = ch_data_column_names[self.reference_dimension]
            assert data_column_name_for_plot == self.variable.df_label, \
                (data_column_name_for_plot, self.variable.df_label, ch_data_column_names)

            for template in mc_channel.tempaltes:
                # TODO: Combine templates which are part of one component?

                # TODO: Get fitted template bin count (maybe reshape to get required multidimensional np.ndarray)
                # TODO: Check if template bin count has expected shape regarding the dimensions!
                # TODO: Do the same for the bin errors!

                # TODO: Loop over all other dimensions than reference dimension.
                # TODO: Get subsets in bins of all other dimensions and plot reference dimension
                #       -> iterative function to get bins in each other dimension!

                template_bin_count = template.expected_bin_counts()
                template_bin_error_sq = template.expected_bin_counts()

                for _ in range(1):  # TODO: Loop over all other bin-combinations
                    current_bin_count = template_bin_count
                    current_bin_errors_squared = template_bin_error_sq

                    self._add_prebinned_component(
                        label=self._get_mc_label(key=template.process_name, original_label=template.latex_label),
                        histogram_key=mc_histogram_key,
                        bin_count=current_bin_count,
                        original_binning=ch_binning,
                        bin_errors_squared=current_bin_errors_squared,
                        data_column_names=data_column_name_for_plot,
                        hist_type="stepfilled",
                        color=self._get_mc_color(key=template.process_name, original_color=template.color),
                        alpha=1.0
                    )

        assert list(set(self._channel_name_list)) == self._channel_name_list, self._channel_name_list

        channel_label_check_list = copy.copy(self._channel_name_list)
        for data_channel in fit_model.data_channels_to_plot:
            if data_channel.name not in channel_label_check_list:
                raise RuntimeError(f"Encountered channel in data channels of fit model, which is not ")
            else:
                channel_label_check_list.remove(data_channel.name)
            data_histogram_key = f"channel_{data_channel.name}_data"

            ch_binning = data_channel.binning.get_binning_for_one_dimension(dimension=self.reference_dimension)
            self._compare_original_binning_to_variable_binning(original_binning=ch_binning)

            ch_data_column_names = data_channel.data_column_names
            data_column_name_for_plot = ch_data_column_names[self.reference_dimension]
            assert data_column_name_for_plot == self.variable.df_label, \
                (data_column_name_for_plot, self.variable.df_label, ch_data_column_names)

            data_bin_count = data_channel.bin_counts
            data_bin_errors_squared = np.sqrt(data_bin_count)

            # TODO: Loop over all other dimensions than reference dimension.
            for _ in range(1):  # TODO: Loop over all other bin-combinations
                current_bin_count = data_bin_count
                current_bin_errors_squared = data_bin_errors_squared

                self._add_prebinned_component(
                    label=self._get_data_label(),
                    histogram_key=data_histogram_key,
                    bin_count=current_bin_count,
                    original_binning=ch_binning,
                    bin_errors_squared=current_bin_errors_squared,
                    data_column_names=data_column_name_for_plot,
                    hist_type="stepfilled",  # TODO: Define new own hist_type for data plots!
                    color=self._get_data_color(),
                    alpha=1.0
                )

    def add_component(
            self,
            label: str,
            data: DataInputType
    ) -> None:
        raise RuntimeError(f"This method should not be used in the plot class {self.__class__.__name__}!")

    def plot_on(self) -> Tuple[figure.Figure, OneOrTwoAxesType]:
        # TODO
        pass

    @property
    def number_of_channels(self) -> int:
        return len(self._channel_name_list)

    @property
    def reference_dimension(self) -> int:
        return self._reference_dimension

    def _get_channel_label(self, key: str, original_label: Optional[str]) -> Optional[str]:
        if "channel_label_dict" in self._optional_arguments_dict:
            channel_label_dict = self._optional_arguments_dict["channel_label_dict"]
            assert isinstance(channel_label_dict, dict), (channel_label_dict, type(channel_label_dict))
            assert all(isinstance(k, str) for k in channel_label_dict.keys()), list(channel_label_dict.keys())
            assert all(isinstance(v, str) for v in channel_label_dict.values()), list(channel_label_dict.values())
            if key not in channel_label_dict.keys():
                raise KeyError(f"No entry for the key {key} in the provided channel_label_dict!\nchannel_label_dict:"
                               f"\n\t" + "\n\t".join([f"{k}: {v}" for k, v in channel_label_dict.items()]))
            return channel_label_dict[key]
        else:
            return original_label

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

    def _compare_original_binning_to_variable_binning(self, original_binning: Binning) -> None:
        assert original_binning.dimensions == 1, original_binning.dimensions

        assert len(original_binning.num_bins) == 1, original_binning.num_bins
        assert original_binning.num_bins[0] == original_binning.num_bins_total, \
            (original_binning.num_bins, original_binning.num_bins_total)

        assert original_binning.num_bins_total == self.variable.n_bins, \
            (original_binning.num_bins_total, self.variable.n_bins)

        assert len(original_binning.range) == 1, original_binning.range
        assert original_binning.range[0] == self.variable.scope, (original_binning.range, self.variable.scope)

        assert len(original_binning.log_scale_mask) == 1, original_binning.log_scale_mask
        assert original_binning.log_scale_mask[0] == self.variable.use_log_scale, \
            (original_binning.log_scale_mask, self.variable.use_log_scale)
