"""
This package provides
    - The Channel class, which holds all components for one reconstruction channel.
    - A ChannelContainer class, which holds all (one or multiple) Channels to be used in the fit model.
"""

import logging

from typing import Optional, List
from collections.abc import Sequence

from templatefitter.fit_model.component import Component
from templatefitter.binned_distributions.binning import Binning
from templatefitter.fit_model.parameter_handler import ParameterHandler

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["Channel", "ChannelContainer"]


class Channel(Sequence):
    def __init__(
            self,
            params: ParameterHandler,
            name: Optional[str] = None,
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

    def __getitem__(self, i) -> Optional[Component]:
        return self._channel_components[i]

    def __len__(self) -> int:
        return len(self._channel_components)


class ChannelContainer(Sequence):
    def __init__(self, channels: Optional[List[Channel]] = None):
        if channels is None:
            self._channels = []
        else:
            self._check_channels_input(channels=channels)
            self._channels = channels

        super().__init__()

    def add_channel(self, channel) -> int:
        if not isinstance(channel, Channel):
            raise ValueError("You can only add instances of 'Channel' to this container.")

        if self.__len__() > 0:
            if self._channels[0].params is not channel.params:
                raise RuntimeError("The used ParameterHandler instances are not the same!")

        channel_index = self.__len__()
        self._channels.append(channel)
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

        return channel_indices

    @staticmethod
    def _check_channels_input(channels: List[Channel]):
        if not isinstance(channels, list):
            raise ValueError(f"The parameter 'channels' must either be a list of Channels or None!\n"
                             f"You provided an object of type {type(channels)}.")
        if not all(isinstance(c, Channel) for c in channels):
            raise ValueError(f"The parameter 'channels' must either be a list of Channels or None!\n"
                             f"The list you provided contained the following types:\n\t-"
                             + "\n\t-".join([str(type(c)) for c in channels]))
        if not all(c.params is channels[0].params for c in channels):
            raise RuntimeError("The used ParameterHandler instances are not the same!")

    def __getitem__(self, i) -> Optional[Channel]:
        return self._channels[i]

    def __len__(self) -> int:
        return len(self._channels)
