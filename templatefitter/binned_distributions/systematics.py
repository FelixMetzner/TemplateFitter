"""
This class provides a general Systematics class
"""

import copy
import logging
import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from typing import Union, Optional, Tuple, List, Sequence, overload

from templatefitter.binned_distributions.binning import Binning
from templatefitter.binned_distributions.weights import Weights, WeightsInputType

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "SystematicsInfo",
    "SystematicsInfoItem",
    "SystematicsInputType",
]

SystematicsUncertInputType = Union[WeightsInputType, List[WeightsInputType]]
SystematicsFromVarInputType = Tuple[WeightsInputType, SystematicsUncertInputType]
MatrixSystematicsInputType = Union[np.ndarray, pd.Series]
SingleSystematicsInputType = Union[None, MatrixSystematicsInputType, SystematicsFromVarInputType]
MultipleSystematicsInputType = List[SingleSystematicsInputType]
SystematicsInputType = Union[SingleSystematicsInputType, MultipleSystematicsInputType]


class SystematicsInfoItem(ABC):
    def __init__(self) -> None:
        self._sys_type = None  # type: Optional[str]
        self._sys_weight = None  # type: Optional[np.ndarray]
        self._sys_uncert = None  # type: Optional[np.ndarray]
        self._cov_matrix = None  # type: Optional[np.ndarray]

    @abstractmethod
    def get_covariance_matrix(
        self,
        binning: Binning,
        data: Optional[np.ndarray] = None,
        weights: WeightsInputType = None,
    ) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def get_varied_hist(
        self,
        initial_varied_hists: Optional[Tuple[np.ndarray, ...]],
        binning: Binning,
        data: Optional[np.ndarray] = None,
        weights: WeightsInputType = None,
    ) -> Tuple[np.ndarray, ...]:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def get_cov_from_varied_hists(varied_hists) -> np.ndarray:
        raise NotImplementedError()

    def get_weights(
        self,
        weights: WeightsInputType,
    ) -> np.ndarray:
        assert self._sys_weight is not None
        if isinstance(weights, float):
            return np.full_like(self._sys_weight, fill_value=weights)
        elif isinstance(weights, np.ndarray):
            assert len(weights) == len(self._sys_weight), (len(weights), len(self._sys_weight))
            return weights
        elif isinstance(weights, pd.Series):
            assert len(weights) == len(self._sys_weight), (len(weights), len(self._sys_weight))
            return weights.values
        else:
            raise TypeError(
                f"The argument weights must be of type float, pd.Series, np.ndarray "
                f"and cannot be of the provided type, which was '{type(weights).__name__}'"
            )


class SystematicsInfoItemFromCov(SystematicsInfoItem):
    def __init__(
        self,
        cov_matrix: np.ndarray,
    ) -> None:
        super().__init__()
        assert isinstance(cov_matrix, np.ndarray), type(cov_matrix)
        assert len(cov_matrix.shape) == 2, cov_matrix.shape
        assert cov_matrix.shape[0] == cov_matrix.shape[1], cov_matrix.shape

        self._sys_type = "cov_matrix"  # type: str
        self._cov_matrix = cov_matrix  # type: np.ndarray

    def get_covariance_matrix(
        self,
        binning: Binning,
        data: Optional[np.ndarray] = None,
        weights: WeightsInputType = None,
    ) -> np.ndarray:
        assert binning is not None
        assert self._cov_matrix.shape[0] == self._cov_matrix.shape[1], self._cov_matrix.shape
        assert binning.num_bins_total == self._cov_matrix.shape[0], (binning.num_bins_total, self._cov_matrix.shape)
        return self._cov_matrix

    def get_varied_hist(
        self,
        initial_varied_hists: Optional[Tuple[np.ndarray, ...]],
        binning: Binning,
        data: Optional[np.ndarray] = None,
        weights: WeightsInputType = None,
    ) -> Tuple[np.ndarray, ...]:
        raise NotImplementedError("This method is not (yet) supported for systematics provided via covariance matrix.")

    @staticmethod
    def get_cov_from_varied_hists(varied_hists: Tuple[np.ndarray, ...]) -> np.ndarray:
        raise NotImplementedError("This method is not (yet) supported for systematics provided via covariance matrix.")


class SystematicsInfoItemFromUpDown(SystematicsInfoItem):
    def __init__(
        self,
        sys_weight: np.ndarray,
        sys_uncert: np.ndarray,
    ) -> None:
        super().__init__()
        assert isinstance(sys_uncert, np.ndarray), type(sys_uncert)
        assert len(sys_uncert.shape) == 1, sys_uncert.shape
        assert len(sys_weight) == len(sys_uncert), (sys_weight.shape, sys_uncert.shape)

        self._sys_type = "up_down"  # type: str
        self._sys_weight = sys_weight  # type: np.ndarray
        self._sys_uncert = sys_uncert  # type: np.ndarray

    def get_covariance_matrix(
        self,
        binning: Binning,
        data: Optional[np.ndarray] = None,
        weights: WeightsInputType = None,
    ) -> np.ndarray:
        assert binning is not None

        varied_hists = self.get_varied_hist(
            initial_varied_hists=None,
            data=data,
            weights=weights,
            binning=binning,
        )
        covariance_matrix = self.get_cov_from_varied_hists(varied_hists=varied_hists)
        assert len(covariance_matrix.shape) == 2, covariance_matrix.shape
        assert covariance_matrix.shape[0] == covariance_matrix.shape[1] == binning.num_bins_total, (
            covariance_matrix.shape,
            binning.num_bins_total,
        )
        return covariance_matrix

    def get_varied_hist(
        self,
        initial_varied_hists: Optional[Tuple[np.ndarray, ...]],
        binning: Binning,
        data: Optional[np.ndarray] = None,
        weights: WeightsInputType = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert data is not None
        assert binning is not None

        assert len(self._sys_weight) == len(data), (len(self._sys_weight), len(data))

        if initial_varied_hists is None:
            initial_varied_hists = (np.zeros(binning.num_bins_total), np.zeros(binning.num_bins_total))
        assert len(initial_varied_hists) == 2, len(initial_varied_hists)

        _weights = self.get_weights(weights=weights)  # type: np.ndarray
        wc = _weights > 0.0
        weights_up = copy.copy(_weights)
        weights_up[wc] = _weights[wc] / self._sys_weight[wc] * (self._sys_weight[wc] + self._sys_uncert[wc])
        weights_dw = copy.copy(_weights)
        weights_dw[wc] = _weights[wc] / self._sys_weight[wc] * (self._sys_weight[wc] - self._sys_uncert[wc])

        bins = [np.array(list(edges)) for edges in binning.bin_edges]
        hist_up, _ = np.histogramdd(data, bins=bins, weights=weights_up)
        hist_dw, _ = np.histogramdd(data, bins=bins, weights=weights_dw)
        assert hist_up.shape == hist_dw.shape, (hist_up.shape, hist_dw.shape)

        if binning.dimensions > 1:
            flat_hist_up = hist_up.flatten()
            flat_hist_dw = hist_dw.flatten()
            assert flat_hist_up.shape == flat_hist_dw.shape, (flat_hist_up.shape, flat_hist_dw.shape)
            assert flat_hist_up.shape[0] == binning.num_bins_total, (flat_hist_up.shape, binning.num_bins_total)

            return initial_varied_hists[0] + flat_hist_up, initial_varied_hists[1] + flat_hist_dw
        else:
            return initial_varied_hists[0] + hist_up, initial_varied_hists[1] + hist_dw

    @staticmethod
    def get_cov_from_varied_hists(varied_hists: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        assert len(varied_hists) == 2, len(varied_hists)
        hist_up, hist_dw = varied_hists
        assert hist_up.shape == hist_dw.shape, (hist_up.shape, hist_dw.shape)

        diff_sym = (hist_up - hist_dw) / 2.0
        return np.outer(diff_sym, diff_sym)


class SystematicsInfoItemFromVariation(SystematicsInfoItem):
    def __init__(
        self,
        sys_weight: np.ndarray,
        sys_uncert: np.ndarray,
    ) -> None:
        super().__init__()
        assert isinstance(sys_uncert, np.ndarray), type(sys_uncert)
        assert len(sys_uncert.shape) == 2, sys_uncert.shape
        assert sys_uncert.shape[1] > 1, sys_uncert.shape
        assert len(sys_weight) == len(sys_uncert), (sys_weight.shape, sys_uncert.shape)

        self._sys_type = "variation"  # type: str
        self._sys_weight = sys_weight  # type: np.ndarray
        self._sys_uncert = sys_uncert  # type: np.ndarray

    def number_of_variations(self) -> int:
        return self._sys_uncert.shape[1]

    def get_covariance_matrix(
        self,
        binning: Binning,
        data: Optional[np.ndarray] = None,
        weights: WeightsInputType = None,
    ) -> np.ndarray:
        varied_hists = self.get_varied_hist(initial_varied_hists=None, data=data, weights=weights, binning=binning)
        return self.get_cov_from_varied_hists(varied_hists=varied_hists)

    def get_varied_hist(
        self,
        initial_varied_hists: Optional[Tuple[np.ndarray, ...]],
        binning: Binning,
        data: Optional[np.ndarray] = None,
        weights: WeightsInputType = None,
    ) -> Tuple[np.ndarray, ...]:
        assert data is not None
        assert binning is not None
        assert len(self._sys_weight) == len(data), (len(self._sys_weight), len(data))

        if initial_varied_hists is None:
            initial_varied_hists = tuple([np.zeros(binning.num_bins_total) for _ in range(self.number_of_variations())])
        assert len(initial_varied_hists) == self.number_of_variations(), (
            len(initial_varied_hists),
            self.number_of_variations(),
        )

        _weights = self.get_weights(weights=weights)  # type: np.ndarray

        varied_hists = []  # type: List[np.ndarray]
        for hist_variation, sys_weight_var in zip(initial_varied_hists, self._sys_uncert.T):
            varied_weights = copy.copy(_weights)
            w_cond = _weights > 0.0
            varied_weights[w_cond] = _weights[w_cond] / self._sys_weight[w_cond] * sys_weight_var[w_cond]

            bins = [np.array(list(edges)) for edges in binning.bin_edges]
            varied_hists.append(hist_variation + np.histogramdd(data, bins=bins, weights=varied_weights)[0].flatten())

        assert len(varied_hists) == len(initial_varied_hists), (len(varied_hists), len(initial_varied_hists))
        assert all(len(vh.shape) == 1 for vh in varied_hists), [vh.shape for vh in varied_hists]
        assert all(vh.shape[0] == binning.num_bins_total for vh in varied_hists), (
            [vh.shape for vh in varied_hists],
            binning.num_bins_total,
        )
        return tuple(varied_hists)

    @staticmethod
    def get_cov_from_varied_hists(varied_hists: Tuple[np.ndarray, ...]) -> np.ndarray:
        cov = np.cov(np.column_stack(varied_hists))
        assert cov.shape[0] == cov.shape[1] == len(varied_hists[0]), (cov.shape[0], cov.shape[1], len(varied_hists[0]))
        assert not np.isnan(cov).any()
        return cov


class SystematicsInfo(Sequence):
    def __init__(
        self,
        in_sys: SystematicsInputType = None,
        data: Optional[np.ndarray] = None,
        in_data: Optional[np.ndarray] = None,
        weights: WeightsInputType = None,
    ):
        self._sys_info_list = self._get_sys_info(in_systematics=in_sys, data=data, in_data=in_data, weights=weights)
        super().__init__()

    def _get_sys_info(
        self,
        in_systematics: SystematicsInputType,
        data: np.ndarray,
        in_data: Optional[pd.DataFrame],
        weights: WeightsInputType,
    ) -> Sequence[Optional[SystematicsInfoItem]]:
        if in_systematics is None:
            return []

        # If not None, systematics must be provided as Tuple for one or List of Tuples for multiple.
        if isinstance(in_systematics, np.ndarray):
            return [self._get_sys_info_from_cov_matrix(in_systematics=in_systematics)]
        elif isinstance(in_systematics, tuple):
            return [
                self._get_single_sys_info(
                    in_systematics=in_systematics,
                    data=data,
                    in_data=in_data,
                    weights=weights,
                )
            ]
        elif isinstance(in_systematics, list):
            return self._get_sys_info_from_list(
                in_systematics=in_systematics,
                data=data,
                in_data=in_data,
                weights=weights,
            )
        else:
            raise ValueError(f"Provided systematics has unexpected type {type(in_systematics)}.")

    @staticmethod
    def _get_sys_info_from_cov_matrix(
        in_systematics: SystematicsInputType,
    ) -> SystematicsInfoItem:
        assert isinstance(in_systematics, np.ndarray), type(in_systematics).__name__
        assert len(in_systematics.shape) == 2, len(in_systematics.shape)
        assert in_systematics.shape[0] == in_systematics.shape[1], (in_systematics.shape[0], in_systematics.shape[1])
        return SystematicsInfoItemFromCov(cov_matrix=in_systematics)

    def _get_single_sys_info(
        self,
        in_systematics: SystematicsInputType,
        data: np.ndarray,
        in_data: Optional[pd.DataFrame],
        weights: WeightsInputType,
    ) -> Optional[SystematicsInfoItem]:
        if in_systematics is None:
            return None

        if len(in_systematics) == 1:
            return self._get_sys_info_from_cov_matrix(in_systematics)
        elif len(in_systematics) == 2:
            assert isinstance(weights, (np.ndarray, pd.Series)), type(weights).__name__
            sys_weight = Weights.obtain_weights(
                weight_input=in_systematics[0], data=data, data_input=in_data
            )  # type: np.ndarray
            assert len(sys_weight) == len(data), (len(sys_weight), len(data))
            assert len(sys_weight) == len(weights)
            assert not np.isnan(sys_weight).any()
            assert np.all(sys_weight[weights > 0.0] > 0.0)

            if isinstance(in_systematics[1], list):
                variations = [Weights.obtain_weights(s, data, in_data) for s in in_systematics[1]]
                sys_uncert = np.column_stack([variation for variation in variations])
                assert sys_uncert.shape[1] == len(in_systematics[1]), (sys_uncert.shape, len(in_systematics[1]))
                assert not np.isnan(sys_uncert).any()
                return SystematicsInfoItemFromVariation(sys_weight=sys_weight, sys_uncert=sys_uncert)
            else:
                sys_uncert = Weights.obtain_weights(
                    weight_input=in_systematics[1],
                    data=data,
                    data_input=in_data,
                )
                assert not np.isnan(sys_uncert).any()
                return SystematicsInfoItemFromUpDown(sys_weight=sys_weight, sys_uncert=sys_uncert)
        else:
            raise ValueError(
                f"Systematics must be provided as tuple or list of tuples"
                f"or directly as the respective covariance matrix. "
                f"Each tuple must contain 2 entries!\n"
                f"A provided tuple was of size {len(in_systematics)} != 2."
            )

    def _get_sys_info_from_list(
        self,
        in_systematics: SystematicsInputType,
        data: np.ndarray,
        in_data: Optional[pd.DataFrame],
        weights: WeightsInputType,
    ) -> Sequence[SystematicsInfoItem]:
        assert isinstance(in_systematics, list), type(in_systematics).__name__
        if len(in_systematics) == 0:
            return []

        result = [
            self._get_single_sys_info(
                in_systematics=in_sys,
                data=data,
                in_data=in_data,
                weights=weights,
            )
            for in_sys in in_systematics
        ]  # type: List[Optional[SystematicsInfoItem]]
        return [e for e in result if e is not None]

    @property
    def as_list(self) -> Sequence[Optional[SystematicsInfoItem]]:
        return self._sys_info_list

    @overload
    def __getitem__(self, i: int) -> SystematicsInfoItem:
        ...

    @overload
    def __getitem__(self, s: slice) -> Sequence[Optional[SystematicsInfoItem]]:
        ...

    def __getitem__(
        self,
        i: Union[int, slice],
    ) -> Union[SystematicsInfoItem, None, Sequence[Optional[SystematicsInfoItem]]]:
        if isinstance(i, slice):
            raise Exception("SystematicsInfo disallows slicing")
        return self._sys_info_list[i]

    def __len__(self) -> int:
        return len(self._sys_info_list)
