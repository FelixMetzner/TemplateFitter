"""
This package provides
    - The Channel class, which holds all components for one reconstruction channel.
    - A ChannelContainer class, which holds all (one or multiple) Channels to be used in the fit model.
"""

import logging

from collections import Counter
from collections.abc import Sequence
from typing import Optional, List, Dict, Tuple

from templatefitter.fit_model.template import Template
from templatefitter.fit_model.component import Component
from templatefitter.binned_distributions.binning import Binning, LogScaleInputType
from templatefitter.fit_model.parameter_handler import ParameterHandler, TemplateParameter
from templatefitter.binned_distributions.binned_distribution import BinnedDistribution, DataInputType, \
    DataColumnNamesInput

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["Channel", "ChannelContainer", "DataChannelContainer"]


class Channel(Sequence):
    def __init__(
            self,
            params: ParameterHandler,
            name: Optional[str],
            components: Optional[List[Component]] = None
    ):
        self._channel_components = []
        if components is not None:
            self.add_components(components=components)

        super().__init__()

        self._params = params
        self._name = name
        self._binning = None
        self._channel_index = None

        self._efficiency_parameters = None

    def add_component(self, component: Component) -> int:
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
                raise RuntimeError(f"Inconsistency in data_column_names of\n\tnew component: {new_data_column_names}\n"
                                   f"and\n\tthis channel: {self.data_column_names}")

        component_index = self.__len__()
        self._channel_components.append(component)
        component.component_index = component_index

        return component_index

    def add_components(self, components: List[Component]) -> List[int]:
        self._check_components_input(components=components)

        if self._binning is None:
            self._binning = components[0].binning
        else:
            if not self._binning == components[0].binning:
                raise ValueError("All components of a channel must have the same binning.")

        # This assignment already checks the consistency of data_column_names of the new components.
        new_data_column_names_list = [c.data_column_names for c in components]
        if self.number_of_components > 0:
            all_data_column_names_list = new_data_column_names_list.append(self.data_column_names)
        else:
            all_data_column_names_list = new_data_column_names_list
        if not all(c == all_data_column_names_list[0] if c is not None else c is all_data_column_names_list[0]
                   for c in all_data_column_names_list):
            raise RuntimeError(f"Inconsistency in data_column_names of new components:\n\t-"
                               + "\n\t-".join([f"{c.name}: {c.data_column_names}" for c in components])
                               + "" if self.number_of_components == 0
                               else f"\nand channel's data_column_names: {self.data_column_names}")

        if not all(c.params is self._params for c in components):
            raise RuntimeError("The used ParameterHandler instances are not the same!")

        first_index = self.__len__()
        self._channel_components.extend(components)
        last_index_plus_one = self.__len__()

        component_indices = list(range(first_index, last_index_plus_one))
        for component, component_index in zip(components, component_indices):
            component.component_index = component_index

        return component_indices

    def initialize_parameters(
            self,
            efficiency_parameters: List[TemplateParameter]
    ) -> None:
        if not isinstance(efficiency_parameters, List):
            raise ValueError(f"Expecting list of TemplateParameters as input for for argument 'efficiency_parameters', "
                             f"but you provided an object of type {efficiency_parameters}")

        if not all(isinstance(eff_par, TemplateParameter) for eff_par in efficiency_parameters):
            raise ValueError(f"Argument 'efficiency_parameters' must be a list containing objects of type "
                             f"TemplateParameter!\nYou provided a list containing the types "
                             f"{[type(eff_par) for eff_par in efficiency_parameters]}...")

        if not len(efficiency_parameters) == self.required_efficiency_parameters:
            raise ValueError(f"Argument 'efficiency_parameters' must be a list of {self.required_efficiency_parameters}"
                             f" TemplateParameters, but the provided list has {len(efficiency_parameters)} elements!")

        if not all(p.parameter_type == ParameterHandler.efficiency_parameter_type for p in efficiency_parameters):
            raise ValueError(f"Expected list of TemplateParameters of parameter_type "
                             f"'{ParameterHandler.efficiency_parameter_type}' for argument 'efficiency_parameters', "
                             f"but received the parameter_types {[p.parameter_type for p in efficiency_parameters]}!")

        self._efficiency_parameters = efficiency_parameters
        for template, efficiency_parameter in zip(self.sub_templates, efficiency_parameters):
            template.efficiency_parameter = efficiency_parameter

    @property
    def name(self) -> str:
        return self._name

    @property
    def params(self) -> ParameterHandler:
        return self._params

    @property
    def binning(self) -> Binning:
        return self._binning

    @property
    def components(self) -> List[Component]:
        return self._channel_components

    @property
    def channel_index(self) -> int:
        assert self._channel_index is not None
        return self._channel_index

    @property
    def channel_serial_number(self) -> int:
        # As channels are at the highest level, their index and serial number are the same.
        assert self._channel_index is not None
        return self._channel_index

    @channel_index.setter
    def channel_index(self, index: int) -> None:
        self._parameter_setter_checker(parameter=self._channel_index, parameter_name="channel_index")
        if not isinstance(index, int):
            raise ValueError("Expected integer...")
        self._channel_index = index
        for component in self._channel_components:
            component.channel_index = self._channel_index

    @property
    def data_column_names(self) -> Optional[List[str]]:
        if self.number_of_components == 0:
            return None
        if not all(c.data_column_names == self._channel_components[0].data_column_names
                   if c.data_column_names is not None
                   else c.data_column_names is self._channel_components[0].data_column_names
                   for c in self._channel_components):
            raise RuntimeError(f"Inconsistency in data_column_names of channel {self.name}:\n\t-"
                               + "\n\t-".join([f"{c.name}: {c.data_column_names}" for c in self._channel_components]))
        return self._channel_components[0].data_column_names

    def _parameter_setter_checker(self, parameter, parameter_name):
        if parameter is not None:
            name_info = "" if self.name is None else f" with name '{self.name}'"
            raise RuntimeError(f"Trying to reset {parameter_name} for channel{name_info}.")

    @property
    def number_of_components(self) -> int:
        return self.__len__()

    @property
    def templates(self) -> Tuple[Template, ...]:
        return tuple([tmp for comp in self._channel_components for tmp in comp.sub_templates])

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

    def __getitem__(self, i) -> Optional[Component]:
        return self._channel_components[i]

    def __len__(self) -> int:
        return len(self._channel_components)

    @staticmethod
    def _check_components_input(components: List[Component]):
        if not isinstance(components, list):
            raise ValueError(f"The parameter 'components' must either be a list of Components or None!\n"
                             f"You provided an object of type {type(components)}.")
        if not all(isinstance(c, Component) for c in components):
            raise ValueError(f"The parameter 'components' must either be a list of Components or None!\n"
                             f"The list you provided contained the following types:\n\t-"
                             + "\n\t-".join([str(type(c)) for c in components]))
        if not all(c.binning == components[0].binning for c in components):
            raise ValueError("All components of a channel must have the same binning.")


class ChannelContainer(Sequence):
    def __init__(self, channels: Optional[List[Channel]] = None):
        self._channels = []
        self._channels_mapping = {}
        if channels is not None:
            self.add_channels(channels=channels)

        super().__init__()

    def add_channel(self, channel) -> int:
        if not isinstance(channel, Channel):
            raise ValueError("You can only add instances of 'Channel' to this container.")

        if self.__len__() > 0:
            if self._channels[0].params is not channel.params:
                raise RuntimeError("The used ParameterHandler instances are not the same!")
            self._check_channel_names(channels=[channel])
            self._check_channel_components(channels=[channel])

        channel_index = self.__len__()
        self._channels.append(channel)
        self._channels_mapping.update({channel.name: channel_index})
        channel.channel_index = channel_index

        return channel_index

    def add_channels(self, channels: List[Channel]) -> List[int]:
        self._check_channels_input(channels=channels)

        first_index = self.__len__()
        self._channels.extend(channels)
        last_index_plus_one = self.__len__()

        channel_indices = list(range(first_index, last_index_plus_one))
        for channel, channel_index in zip(channels, channel_indices):
            channel.channel_index = channel_index
            self._channels_mapping.update({channel.name: channel_index})

        return channel_indices

    def _check_channels_input(self, channels: List[Channel]):
        if not isinstance(channels, list):
            raise ValueError(f"The parameter 'channels' must either be a list of Channels or None!\n"
                             f"You provided an object of type {type(channels)}.")
        if not all(isinstance(c, Channel) for c in channels):
            raise ValueError(f"The parameter 'channels' must either be a list of Channels or None!\n"
                             f"The list you provided contained the following types:\n\t-"
                             + "\n\t-".join([str(type(c)) for c in channels]))
        if not all(c.params is channels[0].params for c in channels):
            raise RuntimeError("The used ParameterHandler instances are not the same!")
        self._check_channel_names(channels=channels)
        self._check_channel_components(channels=channels)

    def _check_channel_names(self, channels: List[Channel]) -> None:
        if any(channel.name in self._channels_mapping.keys() for channel in channels):
            exist_names = [c.name for c in channels if c in self._channels_mapping.keys()]
            exist_i = [self._channels_mapping[n] for n in exist_names]
            raise ValueError(f"Trying to add new channel(s) with name(s) {exist_names} to channel container, but"
                             f"but channel(s) with these name(s) are already registered under the indices {exist_i}.")
        if not len(set([c.name for c in channels])) == len([c.name for c in channels]):
            name_duplicates = [name for name, counter in Counter([c.name for c in channels]).items() if counter > 1]
            raise ValueError(f"Trying to add new channels with same name channel container. "
                             f"The respective channel names are {name_duplicates}.")

    def _check_channel_components(self, channels: List[Channel]) -> None:
        assert isinstance(channels, list), type(channels)
        if self.__len__() > 0:
            base_req_fractions = self._channels[0].required_fraction_parameters
            base_fractions_mask = self._channels[0].fractions_mask
        else:
            base_req_fractions = channels[0].required_fraction_parameters
            base_fractions_mask = channels[0].fractions_mask

        if not all(c.required_fraction_parameters == base_req_fractions for c in channels):
            raise ValueError("You are trying to add channels with different numbers of fraction parameters as "
                             "required due to their components.\n"
                             + (f"The current channels have required_fraction_parameters = \n\t"
                                f"{self._channels[0].required_fraction_parameters}\n" if self.__len__() > 0 else "\n")
                             + "The new channel(s) you are trying to add have required_fraction_parameters = \n\t"
                             + "\n\t".join([f"{c.required_fraction_parameters}" for c in channels])
                             )
        if not all(c.fractions_mask == base_fractions_mask for c in channels):
            raise ValueError("You are trying to add channels with different fraction masks given by their components.\n"
                             + (f"The current channels have the fraction mask = \n\t"
                                f"{self._channels[0].fractions_mask}\n" if self.__len__() > 0 else "\n")
                             + "The new channel(s) you are trying to add have the fraction masks = \n\t"
                             + "\n\t".join([f"{c.fractions_mask}" for c in channels])
                             )

    def _check_number_of_components(self, new_channels: List[Channel]) -> None:
        assert isinstance(new_channels, list), type(new_channels)
        assert len(new_channels) > 0, len(new_channels)
        if len(new_channels) > 1:
            if not all(c.total_number_of_templates == c[0].total_number_of_templates for c in new_channels):
                raise ValueError("Trying to add channels with different numbers of templates:\n\t-"
                                 + "\n\t-".join([f"{c.name}: {c.total_number_of_templates}" for c in new_channels]))

        if self.__len__() > 0:
            new_total_number_of_templates = new_channels[0].total_number_of_templates
            if not all(c.total_number_of_templates == new_total_number_of_templates for c in self._channels):
                raise ValueError("Trying to add new channels with different numbers of templates than for "
                                 "the ones already added. Current channels:\n\t-"
                                 + "\n\t-".join([f"{c.name}: {c.total_number_of_templates}" for c in self._channels])
                                 + "\nNew channels:\n\t-"
                                 + "\n\t-".join([f"{c.name}: {c.total_number_of_templates}" for c in new_channels]))

    @property
    def channel_mapping(self) -> Dict[str, int]:
        assert len(self.channel_mapping) == len(self._channels), (len(self.channel_mapping), len(self._channels))
        return self._channels_mapping

    def get_channel_by_name(self, name: str) -> Channel:
        if name not in self._channels_mapping.keys():
            raise ValueError(f"A channel with the name {name} is not registered in the ChannelContainer.")
        return self._channels[self.channel_mapping[name]]

    def __getitem__(self, i) -> Optional[Channel]:
        return self._channels[i]

    def __len__(self) -> int:
        return len(self._channels)


class DataChannelContainer(Sequence):
    def __init__(
            self,
            channel_names: Optional[List[str]] = None,
            channel_data: Optional[List[DataInputType]] = None,
            binning: Optional[List[Binning]] = None,
            column_names: Optional[Tuple[DataColumnNamesInput]] = None
    ):
        self._channel_distributions = []  # type: List[BinnedDistribution]
        self._channels_mapping = {}  # type: Dict[str, int]
        if channel_names is not None:
            self.add_channels(
                channel_names=channel_names,
                channel_data=channel_data,
                binning=binning,
                column_names=column_names
            )

        super().__init__()

    def add_channels(
            self,
            channel_names: List[str],
            channel_data: List[DataInputType],
            binning: List[Binning],
            column_names: Tuple[DataColumnNamesInput]
    ) -> List[int]:
        if not len(channel_names) == len(channel_data):
            raise ValueError(f"You must provide a channel_name for each channel_data set, "
                             f"but provided {len(channel_names)} names and {len(channel_data)} data sets.")
        if not len(channel_data) == len(binning):
            raise ValueError(f"You must provide a binning definition for each channel_data set, "
                             f"but provided {len(binning)} Binning objects and {len(channel_data)} data sets.")
        if not len(channel_data) == len(column_names):
            raise ValueError(f"You must provide a set of column_names for each channel_data set, "
                             f"but provided {len(column_names)} column_name sets and {len(channel_data)} data sets.")
        if not all(name not in self._channels_mapping.keys() for name in channel_names):
            raise ValueError(f"Data channels with the name(s) "
                             f"{[c for c in channel_names if c in self._channels_mapping]} have already been "
                             f"registered to the DataChannelContainer with the indices "
                             f"{[self._channels_mapping[c] for c in channel_names if c in self._channels_mapping]}.")
        if not len(set(channel_names)) == len(channel_names):
            raise ValueError(f"The channel_names must be unique, but the provided list is not:\n\t{channel_names}")

        channel_indices = []

        for name, data, binning, cols in zip(channel_names, channel_data, binning, column_names):
            channel_index = self.add_channel(
                channel_name=name,
                channel_data=data,
                binning=binning,
                column_names=cols
            )
            channel_indices.append(channel_index)
        assert len(set(channel_indices)) == len(channel_indices), channel_indices
        assert len(channel_names) == len(channel_indices), (len(channel_names), len(channel_indices))

        return channel_indices

    def add_channel(
            self,
            channel_name: str,
            channel_data: DataInputType,
            binning: Binning,
            column_names: DataColumnNamesInput,
            log_scale_mask: LogScaleInputType = False
    ) -> int:
        if channel_name in self._channels_mapping.keys():
            raise RuntimeError(f"Trying to add channel with name '{channel_name}' that is already assigned to the "
                               f"{self._channels_mapping[channel_name]}th channel in the DataChannelContainer.")

        channel_index = self.__len__()
        channel_distribution = BinnedDistribution(
            bins=binning.bin_edges,
            dimensions=binning.dimensions,
            scope=binning.range,
            name=f"data_channel_{channel_index}_{channel_name}",
            data=channel_data,
            weights=None,
            systematics=None,
            data_column_names=column_names,
            log_scale_mask=log_scale_mask
        )

        self._channel_distributions.append(channel_distribution)
        self._channels_mapping.update({channel_name: channel_index})

        return channel_index

    @property
    def data_channel_names(self) -> List[str]:
        return [str(name) for name in self._channels_mapping.keys()]

    @property
    def is_empty(self) -> bool:
        return len(self._channel_distributions) == 0

    def __getitem__(self, i) -> Optional[BinnedDistribution]:
        return self._channel_distributions[i]

    def __len__(self) -> int:
        return len(self._channel_distributions)
