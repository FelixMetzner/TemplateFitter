"""
This package provides
    - The DataChannel classes, which hold the data to which the model shall be fitted to.
    - A DataChannelContainer class, which holds all (one or multiple) DataChannels to be used in the fit model.
    - A ModelDataChannels subclass which needs some information about the fit model to be used
"""

import logging
import numpy as np

from abc import ABC
from typing import Optional, Union, List, Dict, Tuple, Sequence, overload

from templatefitter.fit_model.utility import check_bin_count_shape
from templatefitter.binned_distributions.weights import Weights, WeightsInputType
from templatefitter.binned_distributions.binning import Binning, LogScaleInputType, BinsInputType, ScopeInputType
from templatefitter.binned_distributions.binned_distribution import (
    BinnedDistribution,
    BinnedDistributionFromData,
    BinnedDistributionFromHistogram,
    DataInputType,
    DataColumnNamesInput,
)

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "DataChannelContainer",
    "ModelDataChannels",
]


class DataChannel(ABC):
    def __init__(
        self,
        from_data: bool,
        rounded: bool,
        data_column_names: DataColumnNamesInput = None,
    ) -> None:
        self._from_data = from_data  # type: bool
        self._rounded = rounded  # type: bool

        _columns = None  # type: Optional[List[str]]
        if isinstance(data_column_names, str):
            _columns = [data_column_names]
        else:
            _columns = data_column_names
        self._data_column_names = _columns  # type: Optional[List[str]]

    @property
    def is_from_data(self) -> bool:
        return self._from_data

    @property
    def is_rounded(self) -> bool:
        return self._rounded

    @property
    def data_column_names(self) -> Optional[List[str]]:
        return self._data_column_names


class DataChannelFromData(BinnedDistributionFromData, DataChannel):
    def __init__(
        self,
        bins: BinsInputType,
        dimensions: int,
        scope: ScopeInputType = None,
        log_scale_mask: LogScaleInputType = False,
        name: Optional[str] = None,
        data: Optional[DataInputType] = None,
        weights: WeightsInputType = None,
        data_column_names: DataColumnNamesInput = None,
    ) -> None:
        Weights.check_input_type(weight_input=weights)

        self._rounded = False if weights is None else True  # type: bool

        super().__init__(
            bins=bins,
            dimensions=dimensions,
            scope=scope,
            log_scale_mask=log_scale_mask,
            name=name,
            data=data,
            weights=weights,
            systematics=None,
            data_column_names=data_column_names,
        )

        # Some checks just for fun... and maybe to avoid screw ups when refactoring...
        cls1 = BinnedDistributionFromData.__module__ + "." + BinnedDistributionFromData.__name__
        cls2 = BinnedDistribution.__module__ + "." + BinnedDistribution.__name__
        cls3 = DataChannel.__module__ + "." + DataChannel.__name__

        mro = tuple(f"{c.__module__}.{c.__name__}" for c in self.__class__.__mro__)
        assert mro.index(cls1) < mro.index(cls2) < mro.index(cls3), mro

        self._from_data = True  # type: bool

    @property
    def requires_rounding_due_to_weights(self) -> bool:
        return self.is_rounded

    @property
    def bin_counts(self) -> Union[None, np.ndarray]:
        """ The actual bin counts of the binned distribution; rounded up, if required """
        if self.requires_rounding_due_to_weights:
            return np.ceil(self._bin_counts)

        return self._bin_counts

    @property
    def bin_errors_sq(self) -> Union[None, np.ndarray]:
        """ The squared errors on the bin counts of the binned distribution """
        if self.requires_rounding_due_to_weights:
            return np.ceil(self._bin_counts)

        return self._bin_errors_sq

    @property
    def bin_errors(self) -> Union[None, np.ndarray]:
        if self.bin_errors_sq is None:
            return None
        return np.sqrt(self.bin_errors_sq)


class DataChannelFromHistogram(BinnedDistributionFromHistogram, DataChannel):
    def __init__(
        self,
        bins: BinsInputType,
        dimensions: int,
        scope: ScopeInputType = None,
        log_scale_mask: LogScaleInputType = False,
        name: Optional[str] = None,
        data: Optional[DataInputType] = None,
        data_column_names: DataColumnNamesInput = None,
        round_bin_count: bool = True,
    ) -> None:
        super().__init__(
            bins=bins,
            dimensions=dimensions,
            scope=scope,
            log_scale_mask=log_scale_mask,
            name=name,
            data=data,
            bin_errors_squared=data,
        )

        self._from_data = False  # type: bool
        self._rounded = round_bin_count  # type: bool

        _columns = None  # type: Optional[List[str]]
        if isinstance(data_column_names, str):
            _columns = [data_column_names]
        else:
            _columns = data_column_names
        self._data_column_names = _columns  # type: Optional[List[str]]

        # Some checks just for fun... and maybe to avoid screw ups when refactoring...
        cls1 = BinnedDistributionFromHistogram.__module__ + "." + BinnedDistributionFromHistogram.__name__
        cls2 = BinnedDistribution.__module__ + "." + BinnedDistribution.__name__
        cls3 = DataChannel.__module__ + "." + DataChannel.__name__

        mro = tuple(f"{c.__module__}.{c.__name__}" for c in self.__class__.__mro__)
        assert mro.index(cls1) < mro.index(cls2) < mro.index(cls3), mro

    @property
    def requires_rounding_due_to_weights(self) -> bool:
        return False


class DataChannelContainer(Sequence):
    data_channel_name_prefix = "data_channel"  # type: str

    def __init__(
        self,
        channel_names: Optional[List[str]] = None,
        channel_data: Optional[List[DataInputType]] = None,
        binning: Optional[List[Binning]] = None,
        column_names: Optional[Tuple[DataColumnNamesInput]] = None,
        from_data: Optional[bool] = None,
    ) -> None:
        self._channel_distributions = []  # type: List[Union[DataChannelFromData, DataChannelFromHistogram]]
        self._channels_mapping = {}  # type: Dict[str, int]
        if channel_names is not None:
            assert from_data is not None, "Parameter from_data must be set if DataChannelContainer is filled directly!"
            assert channel_data is not None, "Data must be provided if DataChannelContainer is filled directly!"
            assert column_names is not None, "Column names must be provided if DataChannelContainer is filled directly!"
            assert binning is not None, "Binning must be provided if DataChannelContainer is filled directly!"
            self.add_channels(
                channel_names=channel_names,
                channel_data=channel_data,
                from_data=from_data,
                binning=binning,
                column_names=column_names,
            )

        super().__init__()

    def add_channels(
        self,
        channel_names: List[str],
        channel_data: List[DataInputType],
        from_data: bool,
        binning: List[Binning],
        column_names: Tuple[DataColumnNamesInput],
        channel_weights: Optional[List[WeightsInputType]] = None,
    ) -> List[int]:
        if not len(channel_names) == len(channel_data):
            raise ValueError(
                f"You must provide a channel_name for each channel_data set, "
                f"but provided {len(channel_names)} names and {len(channel_data)} data sets."
            )
        if not len(channel_data) == len(binning):
            raise ValueError(
                f"You must provide a binning definition for each channel_data set, "
                f"but provided {len(binning)} Binning objects and {len(channel_data)} data sets."
            )
        if not len(channel_data) == len(column_names):
            raise ValueError(
                f"You must provide a set of column_names for each channel_data set, "
                f"but provided {len(column_names)} column_name and {len(channel_data)} data sets."
            )
        if not ((channel_weights is None) or (len(channel_data) == len(channel_weights))):
            raise ValueError(
                f"You must provide a set of channel_weights for each channel_data set or "
                f"set channel_weights to None, but you provided {len(channel_weights)} channel_weights "
                f"and {len(channel_data)} data sets."
            )
        if not all(name not in self._channels_mapping.keys() for name in channel_names):
            raise ValueError(
                f"Data channels with the name(s) "
                f"{[c for c in channel_names if c in self._channels_mapping]} have already been "
                f"registered to the DataChannelContainer with the indices "
                f"{[self._channels_mapping[c] for c in channel_names if c in self._channels_mapping]}."
            )
        if not len(set(channel_names)) == len(channel_names):
            raise ValueError(f"The channel_names must be unique, but the provided list is not:\n\t{channel_names}")

        if channel_weights is None:
            ch_weights = [None for _ in range(len(channel_names))]  # type: List[WeightsInputType]
        else:
            ch_weights = channel_weights

        channel_indices = []  # type: List[int]

        for name, data, weights, _binning, cols in zip(channel_names, channel_data, ch_weights, binning, column_names):
            channel_index = self.add_channel(
                channel_name=name,
                channel_data=data,
                from_data=from_data,
                binning=_binning,
                column_names=cols,
                channel_weights=weights,
            )
            channel_indices.append(channel_index)
        assert len(set(channel_indices)) == len(channel_indices), channel_indices
        assert len(channel_names) == len(channel_indices), (len(channel_names), len(channel_indices))

        return channel_indices

    def add_channel(
        self,
        channel_name: str,
        channel_data: DataInputType,
        from_data: bool,
        binning: Binning,
        column_names: DataColumnNamesInput,
        channel_weights: Optional[WeightsInputType] = None,
        log_scale_mask: LogScaleInputType = False,
    ) -> int:
        if channel_name in self._channels_mapping.keys():
            raise RuntimeError(
                f"Trying to add channel with name '{channel_name}' that is already assigned to the "
                f"{self._channels_mapping[channel_name]}th channel in the DataChannelContainer."
            )

        channel_index = len(self)

        if from_data:
            channel_distribution = DataChannelFromData(
                bins=binning.bin_edges,
                dimensions=binning.dimensions,
                scope=binning.range,
                name=self._create_data_channel_name(base_channel_name=channel_name, channel_index=channel_index),
                data=channel_data,
                weights=channel_weights,
                data_column_names=column_names,
                log_scale_mask=log_scale_mask,
            )  # type: Union[DataChannelFromData, DataChannelFromHistogram]
        else:
            assert channel_weights is None, channel_weights
            channel_distribution = DataChannelFromHistogram(
                bins=binning.bin_edges,
                dimensions=binning.dimensions,
                scope=binning.range,
                name=self._create_data_channel_name(base_channel_name=channel_name, channel_index=channel_index),
                data=channel_data,
                data_column_names=column_names,
                log_scale_mask=log_scale_mask,
                round_bin_count=True,
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

    @property
    def requires_rounding_due_to_weights(self) -> Tuple[bool, ...]:
        return tuple([ch.requires_rounding_due_to_weights for ch in self._channel_distributions])

    def _create_data_channel_name(
        self,
        base_channel_name: str,
        channel_index: int,
    ) -> str:
        return f"{self.data_channel_name_prefix}_{channel_index}_{base_channel_name}"

    @staticmethod
    def get_base_channel_name(
        data_channel_name: str,
    ) -> str:
        split_res = data_channel_name.split(f"{DataChannelContainer.data_channel_name_prefix}_", 1)
        assert not split_res[0], split_res
        assert split_res[1], split_res
        split_res = split_res[1].split("_", 1)
        assert split_res[0].isdigit(), split_res
        return split_res[1]

    def get_channel_by_name(
        self,
        name: str,
    ) -> Union[DataChannelFromData, DataChannelFromHistogram]:
        if name not in self.data_channel_names:
            raise KeyError(
                f"The DataChannel of the name '{name}' is not known.\n"
                f"Available channel names are: {self.data_channel_names}"
            )

        return self._channel_distributions[self._channels_mapping[name]]

    @overload
    def __getitem__(self, i: int) -> Optional[BinnedDistribution]:
        ...

    @overload
    def __getitem__(self, s: slice) -> Sequence[Optional[BinnedDistribution]]:
        ...

    def __getitem__(
        self,
        i: Union[int, slice],
    ) -> Union[BinnedDistribution, None, Sequence[Optional[BinnedDistribution]]]:
        if isinstance(i, slice):
            raise Exception("DataChannelContainer disallows slicing")
        return self._channel_distributions[i]

    def __len__(self) -> int:
        return len(self._channel_distributions)


class ModelDataChannels(DataChannelContainer):
    def __init__(
        self,
        data_channel_names: Optional[List[str]] = None,
        channel_data: Optional[List[DataInputType]] = None,
        data_binning: Optional[List[Binning]] = None,
        column_names: Optional[Tuple[DataColumnNamesInput]] = None,
        from_data: Optional[bool] = None,
        model_channel_names: Optional[List[str]] = None,
        model_binning: Optional[List[Binning]] = None,
    ) -> None:

        self._model_channel_names = model_channel_names  # type: Optional[List[str]]
        self._model_binning = model_binning  # type: Optional[List[Binning]]

        if (self._model_channel_names is None) ^ (self._model_binning is None):
            raise ValueError("Either both self.model_channel_names and self._model_binning must be given or neither.")

        self._data_bin_counts = None  # type: Optional[np.ndarray]
        self._data_bin_count_checked = False  # type: bool
        self._data_stat_errors_sq = None  # type: Optional[np.ndarray]
        self._data_stat_errors_checked = False  # type: bool
        super().__init__(data_channel_names, channel_data, data_binning, column_names, from_data)

    def get_data_bin_count_matrix(self) -> np.array:

        flat_data_bin_counts = [data_channel.bin_counts.flatten() for data_channel in self]
        if self._model_binning is not None:
            flat_data_bin_counts = self._apply_padding_to_data_bin_count(bin_counts_per_channel=flat_data_bin_counts)

        self._data_bin_counts = np.stack(flat_data_bin_counts)
        return self._data_bin_counts

    def get_squared_data_stat_errors(self) -> np.ndarray:
        if self._data_stat_errors_sq is not None:
            return self._data_stat_errors_sq

        flat_data_stat_errors_sq = [data_channel.bin_errors_sq.flatten() for data_channel in self]
        if self._model_binning is not None:
            flat_data_stat_errors_sq = self._apply_padding_to_data_bin_count(
                bin_counts_per_channel=flat_data_stat_errors_sq,
            )
        data_stat_errors_sq = np.stack(flat_data_stat_errors_sq)

        if not self._data_stat_errors_checked:
            assert data_stat_errors_sq.shape == self.get_data_bin_count_matrix().shape, (
                data_stat_errors_sq.shape,
                self.get_data_bin_count_matrix().shape,
            )
            self._data_stat_errors_checked = True

        self._data_stat_errors_sq = data_stat_errors_sq
        return data_stat_errors_sq

    def _apply_padding_to_data_bin_count(self, bin_counts_per_channel: List[np.ndarray]) -> List[np.ndarray]:
        if self._model_binning is None or self._model_channel_names is None:
            raise RuntimeError(
                "Padding only makes sense if a separate model binning and model channel names are"
                " specified via the corresponding parameters."
            )

        if not self._data_bin_count_checked:
            assert all(len(bc.shape) == 1 for bc in bin_counts_per_channel), [bc.shape for bc in bin_counts_per_channel]
            assert all(
                len(bc) == b.num_bins_total for bc, b in zip(bin_counts_per_channel, self._model_binning)
            ), "\n".join(
                [
                    f"{chname}: data_bins = {len(bc)} --- template_bins = {b.num_bins_total}"
                    for chname, bc, b in zip(self._model_channel_names, bin_counts_per_channel, self._model_binning)
                ]
            )

        max_n_bins = max([len(bc) for bc in bin_counts_per_channel])
        max_number_of_model_bins_flattened = max(ch_binning.num_bins_total for ch_binning in self._model_binning)

        if not self._data_bin_count_checked:
            assert max_n_bins == max_number_of_model_bins_flattened, (max_n_bins, max_number_of_model_bins_flattened)

        if all(len(bc) == max_n_bins for bc in bin_counts_per_channel):
            return bin_counts_per_channel
        else:
            pad_widths = [(0, max_n_bins - chbin.num_bins_total) for chbin in self._model_binning]
            return [
                np.pad(bc, pad_width=pad_width, mode="constant", constant_values=0)
                for bc, pad_width in zip(bin_counts_per_channel, pad_widths)
            ]

    def get_flattened_data_bin_counts(self, number_of_channels: int, max_number_of_model_bins: int) -> np.array:
        if self._data_bin_counts is not None:
            return self._data_bin_counts

        flat_data_bin_counts = [data_channel.bin_counts.flatten() for data_channel in self]
        if self._model_binning is not None:
            flat_data_bin_counts = self._apply_padding_to_data_bin_count(bin_counts_per_channel=flat_data_bin_counts)
        data_bin_count_matrix = np.stack(flat_data_bin_counts)

        if not self._data_bin_count_checked:
            check_bin_count_shape(
                bin_count=data_bin_count_matrix,
                number_of_channels=number_of_channels,
                max_number_of_bins=max_number_of_model_bins,
                where="get_flattened_data_bin_counts",
            )
            self._data_bin_count_checked = True

        self._data_bin_counts = data_bin_count_matrix
        return data_bin_count_matrix
