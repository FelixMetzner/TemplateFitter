"""
This package provides
    - The Channel class, which holds all components for one reconstruction channel.
    - A ChannelContainer class, which holds all (one or multiple) Channels to be used in the fit model.
"""

import logging

import numpy as np
from collections import Counter
from typing import Union, Optional, List, Dict, Tuple, Sequence, Any, overload

from templatefitter.fit_model.template import Template
from templatefitter.fit_model.component import Component
from templatefitter.binned_distributions.binning import Binning
from templatefitter.fit_model.parameter_handler import ParameterHandler, TemplateParameter

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "Channel",
    "ChannelContainer",
    "ModelChannels",
]


class Channel(Sequence):
    def __init__(
        self,
        params: ParameterHandler,
        name: str,
        latex_label: Optional[str] = None,
        components: Optional[List[Component]] = None,
        plot_order: Optional[Tuple[str, ...]] = None,
    ) -> None:
        self._channel_components = []  # type: List[Component]
        super().__init__()

        self._params = params  # type: ParameterHandler
        self._name = name  # type: str
        self._binning = None  # type: Optional[Binning]
        self._channel_index = None  # type: Optional[int]

        self._plot_order = plot_order  # type: Optional[Tuple[str, ...]]

        if latex_label is None:
            self._latex_label = name  # type str
        else:
            self._latex_label = latex_label

        self._efficiency_parameters = None  # type: Optional[List[TemplateParameter]]

        if components is not None:
            self.add_components(components=components)

    def add_component(
        self,
        component: Component,
    ) -> int:
        if not isinstance(component, Component):
            raise ValueError("You can only add instances of 'Component' to this container.")

        if self._binning is None:
            self._binning = component.binning
        else:
            if not self._binning == component.binning:
                raise ValueError("All components of a channel must have the same binning.")

        if component.params is not self._params:
            raise RuntimeError("The used ParameterHandler instances are not the same!")

        # This assignment already checks the consistency of data_column_names of the new component.
        new_data_column_names = component.data_column_names
        if self.number_of_components > 0:
            if not (new_data_column_names is self.data_column_names or new_data_column_names == self.data_column_names):
                raise RuntimeError(
                    f"Inconsistency in data_column_names of\n\tnew component: {new_data_column_names}\n"
                    f"and\n\tthis channel: {self.data_column_names}"
                )

        component_index = len(self)
        self._channel_components.append(component)
        component.component_index = component_index

        return component_index

    def add_components(
        self,
        components: List[Component],
    ) -> List[int]:
        self._check_components_input(components=components)

        if self._binning is None:
            self._binning = components[0].binning
        else:
            if not self._binning == components[0].binning:
                raise ValueError("All components of a channel must have the same binning.")

        new_data_column_names_list = [c.data_column_names for c in components]  # type: List[Optional[List[str]]]
        assert all(
            c == new_data_column_names_list[0] if new_data_column_names_list[0] is not None else c is None
            for c in new_data_column_names_list
        ), new_data_column_names_list

        if self.number_of_components > 0:
            if not all(
                c == self.data_column_names if self.data_column_names is not None else c is None
                for c in new_data_column_names_list
            ):
                raise RuntimeError(
                    "Inconsistency in data_column_names of new components:\n\t-"
                    + "\n\t-".join([f"{c.name}: {c.data_column_names}" for c in components])
                    + ""
                    if self.number_of_components == 0
                    else f"\nand channel's data_column_names: {self.data_column_names}"
                )

        if not all(c.params is self._params for c in components):
            raise RuntimeError("The used ParameterHandler instances are not the same!")

        first_index = len(self)
        self._channel_components.extend(components)
        last_index_plus_one = len(self)

        component_indices = list(range(first_index, last_index_plus_one))
        for component, component_index in zip(components, component_indices):
            component.component_index = component_index

        return component_indices

    def initialize_parameters(
        self,
        efficiency_parameters: List[TemplateParameter],
    ) -> None:
        if not isinstance(efficiency_parameters, List):
            raise ValueError(
                f"Expecting list of TemplateParameters as input for for argument 'efficiency_parameters', "
                f"but you provided an object of type {efficiency_parameters}"
            )

        if not all(isinstance(eff_par, TemplateParameter) for eff_par in efficiency_parameters):
            raise ValueError(
                f"Argument 'efficiency_parameters' must be a list containing objects of type "
                f"TemplateParameter!\nYou provided a list containing the types "
                f"{[type(eff_par) for eff_par in efficiency_parameters]}..."
            )

        if not len(efficiency_parameters) == self.required_efficiency_parameters:
            raise ValueError(
                f"Argument 'efficiency_parameters' must be a list of {self.required_efficiency_parameters}"
                f" TemplateParameters, but the provided list has {len(efficiency_parameters)} elements!"
            )

        if not all(p.parameter_type == ParameterHandler.efficiency_parameter_type for p in efficiency_parameters):
            raise ValueError(
                f"Expected list of TemplateParameters of parameter_type "
                f"'{ParameterHandler.efficiency_parameter_type}' for argument 'efficiency_parameters', "
                f"but received the parameter_types {[p.parameter_type for p in efficiency_parameters]}!"
            )

        self._efficiency_parameters = efficiency_parameters
        for template, efficiency_parameter in zip(self.sub_templates, efficiency_parameters):
            template.efficiency_parameter = efficiency_parameter

    @property
    def name(self) -> str:
        return self._name

    @property
    def latex_label(self) -> str:
        return self._latex_label

    @property
    def params(self) -> ParameterHandler:
        return self._params

    @property
    def binning(self) -> Binning:
        assert self._binning is not None
        return self._binning

    @property
    def components(self) -> List[Component]:
        return self._channel_components

    @property
    def channel_index(self) -> int:
        assert self._channel_index is not None
        return self._channel_index

    @channel_index.setter
    def channel_index(
        self,
        index: int,
    ) -> None:
        self._parameter_setter_checker(parameter=self._channel_index, parameter_name="channel_index")
        if not isinstance(index, int):
            raise ValueError("Expected integer...")
        self._channel_index = index
        for component in self._channel_components:
            component.channel_index = self._channel_index

    @property
    def channel_serial_number(self) -> int:
        # As channels are at the highest level, their index and serial number are the same.
        return self.channel_index

    @property
    def data_column_names(self) -> Optional[List[str]]:
        if self.number_of_components == 0:
            return None
        if not all(
            c.data_column_names == self._channel_components[0].data_column_names
            if c.data_column_names is not None
            else c.data_column_names is self._channel_components[0].data_column_names
            for c in self._channel_components
        ):
            raise RuntimeError(
                f"Inconsistency in data_column_names of channel {self.name}:\n\t-"
                + "\n\t-".join([f"{c.name}: {c.data_column_names}" for c in self._channel_components])
            )
        return self._channel_components[0].data_column_names

    def _parameter_setter_checker(
        self,
        parameter: Any,
        parameter_name: str,
    ) -> None:
        if parameter is not None:
            name_info = "" if self.name is None else f" with name '{self.name}'"
            raise RuntimeError(f"Trying to reset {parameter_name} for channel{name_info}.")

    @property
    def number_of_components(self) -> int:
        return len(self)

    @property
    def templates(self) -> Tuple[Template, ...]:
        return tuple([tmp for comp in self._channel_components for tmp in comp.sub_templates])

    @property
    def templates_in_plot_order(self) -> Tuple[Template, ...]:
        if self._plot_order is None:
            return tuple(t for t in self.templates if not t.is_irrelevant)
        else:
            template_names = [t.name for t in self.templates if not t.is_irrelevant]  # type: List[str]
            template_map = {t.name: t for t in self.templates if not t.is_irrelevant}  # type: Dict[str, Template]
            assert len(template_map.keys()) == len(template_names), (
                len(template_map.keys()),
                len(template_names),
                list(template_map.keys()),
                template_names,
            )
            assert all(t in template_names for t in self._plot_order), [
                t for t in self._plot_order if t not in template_names
            ]
            return tuple(template_map[tn] for tn in self._plot_order)

    @property
    def process_names(self) -> List[str]:
        return [pn for c in self._channel_components for pn in c.process_names]

    @property
    def process_names_per_component(self) -> List[List[str]]:
        return [c.process_names for c in self._channel_components]

    @property
    def component_names(self) -> List[str]:
        return [c.name for c in self._channel_components]

    @property
    def component_serial_numbers(self) -> Tuple[int, ...]:
        return tuple(c.component_serial_number for c in self._channel_components)

    @property
    def number_of_templates_per_component(self) -> Tuple[int, ...]:
        return tuple(c.number_of_subcomponents for c in self._channel_components)

    @property
    def total_number_of_templates(self) -> int:
        return sum([c.number_of_subcomponents for c in self._channel_components])

    @property
    def template_serial_numbers(self) -> Tuple[int, ...]:
        return tuple(tsn for c in self._channel_components for tsn in c.template_serial_numbers)

    @property
    def sub_templates(self) -> List[Template]:
        return [t for c in self._channel_components for t in c.sub_templates]

    @property
    def required_efficiency_parameters(self) -> int:
        # As efficiencies are different for each template, we need as many efficiency parameters as templates.
        return self.total_number_of_templates

    @property
    def required_fraction_parameters(self) -> Tuple[int, ...]:
        return tuple([c.required_fraction_parameters for c in self._channel_components])

    @property
    def fractions_mask(self) -> Tuple[bool, ...]:
        mask = []
        for c_fractions in self.required_fraction_parameters:
            if c_fractions == 0:
                mask.append(False)
            else:
                mask.extend([True] * c_fractions)
        return tuple(mask)

    @overload
    def __getitem__(self, i: int) -> Optional[Component]:
        ...

    @overload
    def __getitem__(self, s: slice) -> Sequence[Optional[Component]]:
        ...

    def __getitem__(
        self,
        i: Union[int, slice],
    ) -> Union[Component, None, Sequence[Optional[Component]]]:
        if isinstance(i, slice):
            raise Exception("Channel disallows slicing")
        return self._channel_components[i]

    def __len__(self) -> int:
        return len(self._channel_components)

    def __repr__(self) -> str:
        return f"Channel({str(self._channel_components)})"

    @staticmethod
    def _check_components_input(
        components: List[Component],
    ) -> None:
        if not isinstance(components, list):
            raise ValueError(
                f"The parameter 'components' must either be a list of Components or None!\n"
                f"You provided an object of type {type(components)}."
            )
        if not all(isinstance(c, Component) for c in components):
            raise ValueError(
                "The parameter 'components' must either be a list of Components or None!\n"
                "The list you provided contained the following types:\n\t-"
                + "\n\t-".join([str(type(c)) for c in components])
            )
        if not all(c.binning == components[0].binning for c in components):
            raise ValueError("All components of a channel must have the same binning.")


class ChannelContainer(Sequence):
    def __init__(
        self,
        channels: Optional[List[Channel]] = None,
    ) -> None:
        self._channels = []  # type: List[Channel]
        self._channels_mapping = {}  # type: Dict[str, int]
        if channels is not None:
            self.add_channels(channels=channels)

        super().__init__()

    def add_channel(
        self,
        channel: Channel,
    ) -> int:
        if not isinstance(channel, Channel):
            raise ValueError("You can only add instances of 'Channel' to this container.")

        if len(self):
            if self._channels[0].params is not channel.params:
                raise RuntimeError("The used ParameterHandler instances are not the same!")
            self._check_channel_names(channels=[channel])
            self._check_channel_components(channels=[channel])

        channel_index = len(self)
        self._channels.append(channel)
        self._channels_mapping.update({channel.name: channel_index})
        assert len(self._channels_mapping) == len(self._channels), (len(self._channels_mapping), len(self._channels))

        channel.channel_index = channel_index

        return channel_index

    def add_channels(
        self,
        channels: List[Channel],
    ) -> List[int]:
        self._check_channels_input(channels=channels)

        first_index = len(self)
        self._channels.extend(channels)
        last_index_plus_one = len(self)

        channel_indices = list(range(first_index, last_index_plus_one))
        for channel, channel_index in zip(channels, channel_indices):
            channel.channel_index = channel_index
            self._channels_mapping.update({channel.name: channel_index})

        assert len(self._channels_mapping) == len(self._channels), (len(self._channels_mapping), len(self._channels))

        return channel_indices

    def _check_channels_input(
        self,
        channels: List[Channel],
    ) -> None:
        if not isinstance(channels, list):
            raise ValueError(
                f"The parameter 'channels' must either be a list of Channels or None!\n"
                f"You provided an object of type {type(channels)}."
            )
        if not all(isinstance(c, Channel) for c in channels):
            raise ValueError(
                "The parameter 'channels' must either be a list of Channels or None!\n"
                "The list you provided contained the following types:\n\t-"
                + "\n\t-".join([str(type(c)) for c in channels])
            )
        if not all(c.params is channels[0].params for c in channels):
            raise RuntimeError("The used ParameterHandler instances are not the same!")
        self._check_channel_names(channels=channels)
        self._check_channel_components(channels=channels)

    def _check_channel_names(
        self,
        channels: List[Channel],
    ) -> None:
        if any(channel.name in self._channels_mapping.keys() for channel in channels):
            exist_names = [c.name for c in channels if c in self._channels_mapping.keys()]
            exist_i = [self._channels_mapping[n] for n in exist_names]
            raise ValueError(
                f"Trying to add new channel(s) with name(s) {exist_names} to channel container, but"
                f"but channel(s) with these name(s) are already registered under the indices {exist_i}."
            )
        if not len(set([c.name for c in channels])) == len([c.name for c in channels]):
            name_duplicates = [name for name, counter in Counter([c.name for c in channels]).items() if counter > 1]
            raise ValueError(
                f"Trying to add new channels with same name channel container. "
                f"The respective channel names are {name_duplicates}."
            )

    def _check_channel_components(
        self,
        channels: List[Channel],
    ) -> None:
        assert isinstance(channels, list), type(channels)
        if len(self):
            base_req_fractions = self._channels[0].required_fraction_parameters
            base_fractions_mask = self._channels[0].fractions_mask
        else:
            base_req_fractions = channels[0].required_fraction_parameters
            base_fractions_mask = channels[0].fractions_mask

        if not all(c.required_fraction_parameters == base_req_fractions for c in channels):
            raise ValueError(
                "You are trying to add channels with different numbers of fraction parameters as "
                "required due to their components.\n"
                + (
                    f"The current channels have required_fraction_parameters = \n\t"
                    f"{self._channels[0].required_fraction_parameters}\n"
                    if len(self) > 0
                    else "\n"
                )
                + "The new channel(s) you are trying to add have required_fraction_parameters = \n\t"
                + "\n\t".join([f"{c.required_fraction_parameters}" for c in channels])
            )
        if not all(c.fractions_mask == base_fractions_mask for c in channels):
            raise ValueError(
                "You are trying to add channels with different fraction masks given by their components.\n"
                + (
                    f"The current channels have the fraction mask = \n\t" f"{self._channels[0].fractions_mask}\n"
                    if len(self) > 0
                    else "\n"
                )
                + "The new channel(s) you are trying to add have the fraction masks = \n\t"
                + "\n\t".join([f"{c.fractions_mask}" for c in channels])
            )

    def _check_number_of_components(
        self,
        new_channels: List[Channel],
    ) -> None:
        assert isinstance(new_channels, list), type(new_channels)
        assert len(new_channels) > 0, len(new_channels)
        if len(new_channels) > 1:
            if not all(c.total_number_of_templates == new_channels[0].total_number_of_templates for c in new_channels):
                raise ValueError(
                    "Trying to add channels with different numbers of templates:\n\t-"
                    + "\n\t-".join([f"{c.name}: {c.total_number_of_templates}" for c in new_channels])
                )

        if len(self):
            new_total_number_of_templates = new_channels[0].total_number_of_templates
            if not all(c.total_number_of_templates == new_total_number_of_templates for c in self._channels):
                raise ValueError(
                    "Trying to add new channels with different numbers of templates than for "
                    "the ones already added. Current channels:\n\t-"
                    + "\n\t-".join([f"{c.name}: {c.total_number_of_templates}" for c in self._channels])
                    + "\nNew channels:\n\t-"
                    + "\n\t-".join([f"{c.name}: {c.total_number_of_templates}" for c in new_channels])
                )

    @property
    def channel_mapping(self) -> Dict[str, int]:
        return self._channels_mapping

    def get_channel_by_name(
        self,
        name: str,
    ) -> Channel:
        if name not in self._channels_mapping.keys():
            raise ValueError(f"A channel with the name {name} is not registered in the ChannelContainer.")
        return self._channels[self.channel_mapping[name]]

    @overload
    def __getitem__(self, i: int) -> Optional[Channel]:
        ...

    @overload
    def __getitem__(self, s: slice) -> Sequence[Optional[Channel]]:
        ...

    def __getitem__(
        self,
        i: Union[int, slice],
    ) -> Union[Channel, None, Sequence[Optional[Channel]]]:
        if isinstance(i, slice):
            raise Exception("ChannelContainer disallows slicing")
        return self._channels[i]

    def __len__(self) -> int:
        return len(self._channels)

    def __repr__(self) -> str:
        return f"ChannelContainer({str(self._channels)})"


class ModelChannels(ChannelContainer):
    def __init__(
        self,
        channels: Optional[List[Channel]] = None,
    ) -> None:

        self._is_checked = False
        self._template_shapes_checked = False
        self._template_bin_counts = None  # type: Optional[np.ndarray]

        super().__init__(channels)

    @property
    def binning(self) -> Tuple[Binning, ...]:
        return tuple(channel.binning for channel in self)

    @property
    def max_number_of_bins_flattened(self) -> int:
        return max(ch_binning.num_bins_total for ch_binning in self.binning)

    @property
    def number_of_bins_flattened_per_channel(self) -> List[int]:
        return [ch_binning.num_bins_total for ch_binning in self.binning]

    @property
    def number_of_components(self) -> Tuple[int, ...]:
        return tuple(len(channel) for channel in self)

    @property
    def total_number_of_templates(self) -> int:
        return sum([c.total_number_of_templates for c in self])

    @property
    # Number of templates per channel
    def number_of_templates(self) -> Tuple[int, ...]:
        return tuple(sum([comp.number_of_subcomponents for comp in ch.components]) for ch in self)

    @property
    def number_of_independent_templates(self) -> Tuple[int, ...]:
        return tuple(
            sum([1 if comp.shared_yield else comp.number_of_subcomponents for comp in ch.components]) for ch in self
        )

    @property
    def number_of_dependent_templates(self) -> Tuple[int, ...]:
        return tuple([t - it for t, it in zip(self.number_of_templates, self.number_of_independent_templates)])

    @property
    def number_of_expected_independent_yields(self) -> int:
        return max(self.number_of_independent_templates)

    @property
    def min_number_of_independent_yields(self) -> int:
        return min(self.number_of_independent_templates)

    @property
    def number_of_fraction_parameters(self) -> Tuple[int, ...]:
        return tuple(sum([comp.required_fraction_parameters for comp in ch.components]) for ch in self)

    @property
    def template_bin_counts(self) -> np.ndarray:
        if self._template_bin_counts is not None:
            return self._template_bin_counts

        bin_counts_per_channel = [np.stack([tmp.bin_counts.flatten() for tmp in ch.templates]) for ch in self]
        padded_bin_counts_per_channel = self._calc_padding_for_templates(bin_counts_per_channel=bin_counts_per_channel)
        template_bin_counts = np.stack(padded_bin_counts_per_channel)

        if not self._template_shapes_checked:
            self._check_template_shapes(template_bin_counts=template_bin_counts)

        self._template_bin_counts = template_bin_counts
        return self._template_bin_counts

    def _calc_padding_for_templates(
        self,
        bin_counts_per_channel: List[np.ndarray],
    ) -> List[np.ndarray]:
        if not self._template_shapes_checked:
            assert all(
                [bc.shape[1] == ch.binning.num_bins_total for bc, ch in zip(bin_counts_per_channel, self)]
            ), "\t" + "\n\t".join(
                [f"{bc.shape[1]} : {ch.binning.num_bins_total}" for bc, ch in zip(bin_counts_per_channel, self)]
            )

        max_n_bins = max([bc.shape[1] for bc in bin_counts_per_channel])

        if all(bc.shape[1] == max_n_bins for bc in bin_counts_per_channel):
            return bin_counts_per_channel
        else:
            pad_widths = self._pad_widths_per_channel()
            return [
                np.pad(bc, pad_width=pad_width, mode="constant", constant_values=0)
                for bc, pad_width in zip(bin_counts_per_channel, pad_widths)
            ]

    def _pad_widths_per_channel(self) -> List[List[Tuple[int, int]]]:
        max_n_bins = max([ch.binning.num_bins_total for ch in self])
        if not self._is_checked:
            assert max_n_bins == self.max_number_of_bins_flattened, (max_n_bins, self.max_number_of_bins_flattened)
        return [[(0, 0), (0, max_n_bins - ch.binning.num_bins_total)] for ch in self]

    def _check_template_shapes(
        self,
        template_bin_counts: np.ndarray,
    ) -> None:
        # Check order of processes in channels:
        _first_channel = self._channels[0]
        assert isinstance(_first_channel, Channel), type(_first_channel).__name__
        assert all(ch.process_names == _first_channel.process_names for ch in self), [ch.process_names for ch in self]

        # Check shape of template_bin_counts
        assert len(template_bin_counts.shape) == 3, (len(template_bin_counts.shape), template_bin_counts.shape)
        assert template_bin_counts.shape[0] == len(self), (
            template_bin_counts.shape,
            template_bin_counts.shape[0],
            len(self),
        )
        assert all(template_bin_counts.shape[1] == ts_in_ch for ts_in_ch in self.number_of_templates), (
            template_bin_counts.shape,
            template_bin_counts.shape[1],
            [t_ch for t_ch in self.number_of_templates],
        )
        assert template_bin_counts.shape[2] == self.max_number_of_bins_flattened, (
            template_bin_counts.shape,
            template_bin_counts.shape[2],
            self.max_number_of_bins_flattened,
        )

        self._template_shapes_checked = True
