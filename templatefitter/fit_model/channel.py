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
from templatefitter.binned_distributions.binning import Binning
from templatefitter.fit_model.parameter_handler import ParameterHandler, TemplateParameter

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["Channel", "ChannelContainer"]


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

    def _parameter_setter_checker(self, parameter, parameter_name):
        if parameter is not None:
            name_info = "" if self.name is None else f" with name '{self.name}'"
            raise RuntimeError(f"Trying to reset {parameter_name} for channel{name_info}.")

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

    @property
    def number_of_components(self) -> int:
        return self.__len__()

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

    def __getitem__(self, i) -> Optional[Component]:
        return self._channel_components[i]

    def __len__(self) -> int:
        return len(self._channel_components)


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

    def __getitem__(self, i) -> Optional[Channel]:
        return self._channels[i]

    def __len__(self) -> int:
        return len(self._channels)
