"""
This package provides
    - The Channel class, which holds all components for one reconstruction channel.
    - A ChannelContainer class, which holds all (one or multiple) Channels to be used in the fit model.
"""

import logging

from typing import Optional, List
from collections.abc import Sequence

from templatefitter.fit_model.component import Component

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["Channel", "ChannelContainer"]


class Channel(Sequence):
    def __init__(self, components: Optional[List[Component]] = None):
        if components is None:
            self._channel_components = []
        else:
            self._check_components_input(components=components)
            self._channel_components = components

        super().__init__()

    def add_component(self, component: Component) -> int:
        if not isinstance(component, Component):
            raise ValueError("You can only add instances of 'Component' to this container.")

        component_index = self.__len__()
        self._channel_components.append(component)

        return component_index

    def add_components(self, components: List[Component]) -> List[int]:
        self._check_components_input(components=components)

        first_index = self.__len__()
        self._channel_components.extend(components)
        last_index_plus_one = self.__len__()

        return list(range(first_index, last_index_plus_one))

    @staticmethod
    def _check_components_input(components: List[Component]):
        if not isinstance(components, list):
            raise ValueError(f"The parameter 'components' must either be a list of Components or None!\n"
                             f"You provided an object of type {type(components)}.")
        if not all(isinstance(c, Component) for c in components):
            raise ValueError(f"The parameter 'components' must either be a list of Components or None!\n"
                             f"The list you provided contained the following types:\n\t-"
                             + "\n\t-".join([str(type(c)) for c in components]))

    def __getitem__(self, i) -> Optional[Component]:
        return self._channel_components[i]

    def __len__(self) -> int:
        return len(self._channel_components)

    # TODO: Assign fractions, parameters, indices and so on
    # TODO: Ensure, that all components/templates in the channel have the same binning!
    # TODO: Check that every template of a model uses the same ParameterHandler instance!


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

        channel_index = self.__len__()
        self._channels.append(channel)

        return channel_index

    def add_channels(self, channels: List[Channel]) -> List[int]:
        self._check_channels_input(channels=channels)

        first_index = self.__len__()
        self._channels.extend(channels)
        last_index_plus_one = self.__len__()

        return list(range(first_index, last_index_plus_one))

    @staticmethod
    def _check_channels_input(channels: List[Channel]):
        if not isinstance(channels, list):
            raise ValueError(f"The parameter 'channels' must either be a list of Channels or None!\n"
                             f"You provided an object of type {type(channels)}.")
        if not all(isinstance(c, Channel) for c in channels):
            raise ValueError(f"The parameter 'channels' must either be a list of Channels or None!\n"
                             f"The list you provided contained the following types:\n\t-"
                             + "\n\t-".join([str(type(c)) for c in channels]))

    def __getitem__(self, i) -> Optional[Channel]:
        return self._channels[i]

    def __len__(self) -> int:
        return len(self._channels)

    # TODO: Check that every template of a model uses the same ParameterHandler instance!
