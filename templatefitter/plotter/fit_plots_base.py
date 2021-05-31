"""
Plotting tools to illustrate fit results produced with this package
"""
import os
import copy
import logging
import itertools
import numpy as np

from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
from typing import Optional, Union, Tuple, List, Dict, NamedTuple, Generator, Any, Type

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
    "FitPlotBase",
    "SubBinInfos",
    "FitPlotterBase",
]

plot_style.set_matplotlibrc_params()


class FitPlotBase(HistogramPlot, ABC):  # TODO: Implement and use in result and template plots
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


class SubBinInfos(NamedTuple):
    bin_ids: Tuple[int, ...]
    bin_edges: Tuple[Tuple[float, float], ...]


class FitPlotterBase(ABC):  # TODO: Implement and use in result and template plots
    def __init__(
        self,
        variables: Tuple[HistVariable, ...],
        fit_model: FitModel,
        reference_dimension: int = 0,
        fig_size: Tuple[float, float] = (5, 5),
        **kwargs,
    ) -> None:
        self._variables = variables  # type: Tuple[HistVariable, ...]

        self._fit_model = fit_model  # type: FitModel

        self._base_reference_dimension = reference_dimension  # type: int

        self._fig_size = fig_size  # type: Tuple[float, float]
        self._optional_arguments_dict = kwargs  # type: Dict[str, Any]

        self._channel_name_list = []  # type: List[str]
        self._channel_variables_per_dim = {}  # type: Dict[str, Dict[int, HistVariable]]
        self._number_of_bins_per_ch_per_dim = {}  # type: Dict[str, Dict[int, int]]

        self._get_histogram_infos_from_model(fit_model=fit_model)

        self._plotter_class = None  # type: Optional[Type[FitPlotBase]]

    @property
    def plotter_class(self) -> Type[FitPlotBase]:
        assert self._plotter_class is not None
        return self._plotter_class

    def _get_histogram_infos_from_model(
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
    def base_reference_dimension(self) -> int:
        return self._base_reference_dimension

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
    ) -> str:
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

    @staticmethod
    def _get_slices(
        reference_dimension: int,
        sub_bin_info: Optional[SubBinInfos],
    ) -> Tuple[Union[slice, int], ...]:
        if sub_bin_info is None:
            assert reference_dimension == 0
            return tuple([slice(None)])

        assert isinstance(sub_bin_info, SubBinInfos), type(sub_bin_info)
        bins_in_other_dims = sub_bin_info.bin_ids
        assert isinstance(bins_in_other_dims, tuple), type(bins_in_other_dims)
        assert all(isinstance(n_bins, int) for n_bins in bins_in_other_dims), bins_in_other_dims

        slice_list = []  # type: List[Union[slice, int]]
        for dim in range(len(bins_in_other_dims) + 1):
            if dim == reference_dimension:
                slice_list.append(slice(None))
            else:
                slice_list.append(bins_in_other_dims[dim if dim < reference_dimension else dim - 1])

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
