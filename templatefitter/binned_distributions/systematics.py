"""
This class provides a general Systematics class
"""

import copy
import logging
import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Union, Optional, Tuple, List

from templatefitter.binned_distributions.binning import BinEdgesType
from templatefitter.binned_distributions.weights import Weights, WeightsInputType

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["SystematicsInfo", "SystematicsInputType"]

SystematicsUncertInputType = Union[WeightsInputType, List[WeightsInputType]]
SystematicsFromVarInputType = Tuple[WeightsInputType, SystematicsUncertInputType]
MatrixSystematicsInputType = np.ndarray
SingleSystematicsInputType = Union[None, MatrixSystematicsInputType, SystematicsFromVarInputType]
MultipleSystematicsInputType = List[SingleSystematicsInputType]
SystematicsInputType = Union[None, SingleSystematicsInputType, MultipleSystematicsInputType]


# TODO: Conversion from 1-D histograms to n-D necessary!
#  weights, data, bin_edges, etc. have to be handled correctly!
# TODO: Check weights shapes
# TODO: Check bin_edges shapes
# TODO: Check data shapes!


class SystematicsInfoItem(ABC):
    def __init__(self):
        self._sys_type = None
        self._sys_weight = None
        self._sys_uncert = None
        self._cov_matrix = None

    @abstractmethod
    def get_cov(self, data=None, weights=None, bin_edges=None):
        raise NotImplementedError()

    @abstractmethod
    def get_varied_hist(self, initial_varied_hists, data=None, weights=None, bin_edges=None):
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def get_cov_from_varied_hists(varied_hists):
        raise NotImplementedError()


class SystematicsInfoItemFromCov(SystematicsInfoItem):
    def __init__(self, cov_matrix: np.ndarray):
        super().__init__()
        assert isinstance(cov_matrix, np.ndarray), type(cov_matrix)
        assert len(cov_matrix.shape) == 2, cov_matrix.shape
        assert cov_matrix.shape[0] == cov_matrix.shape[1], cov_matrix.shape

        self._sys_type = "cov_matrix"
        self._cov_matrix = cov_matrix

    def get_cov(
            self,
            data: Optional[np.ndarray] = None,
            weights: WeightsInputType = None,
            bin_edges: Optional[BinEdgesType] = None
    ) -> np.ndarray:
        assert bin_edges is not None
        assert len(bin_edges) - 1 == self._cov_matrix.shape[0], (len(bin_edges) - 1, self._cov_matrix.shape[0])
        assert len(bin_edges) - 1 == self._cov_matrix.shape[1], (len(bin_edges) - 1, self._cov_matrix.shape[1])
        return self._cov_matrix

    def get_varied_hist(self, initial_varied_hists, data=None, weights=None, bin_edges=None):
        raise NotImplementedError("This method is not (yet) supported for systematics provided via covariance matrix.")

    @staticmethod
    def get_cov_from_varied_hists(varied_hists):
        raise NotImplementedError("This method is not (yet) supported for systematics provided via covariance matrix.")


class SystematicsInfoItemFromUpDown(SystematicsInfoItem):
    def __init__(self, sys_weight: np.ndarray, sys_uncert: np.ndarray):
        super().__init__()
        self._sys_type = "up_down"
        assert isinstance(sys_uncert, np.ndarray), type(sys_uncert)
        assert len(sys_uncert.shape) == 1, sys_uncert.shape
        assert len(sys_weight) == len(sys_uncert), (sys_weight.shape, sys_uncert.shape)

        self._sys_weight = sys_weight
        self._sys_uncert = sys_uncert

    def get_cov(
            self,
            data: Optional[np.ndarray] = None,
            weights: WeightsInputType = None,
            bin_edges: Optional[BinEdgesType] = None
    ) -> np.ndarray:
        varied_hists = self.get_varied_hist(initial_varied_hists=None, data=data, weights=weights, bin_edges=bin_edges)
        return self.get_cov_from_varied_hists(varied_hists=varied_hists)

    def get_varied_hist(
            self,
            initial_varied_hists,
            data: Optional[np.ndarray] = None,
            weights: WeightsInputType = None,
            bin_edges: Optional[BinEdgesType] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert data is not None
        assert weights is not None
        assert bin_edges is not None
        if initial_varied_hists is None:
            initial_varied_hists = (np.zeros(len(bin_edges) - 1), np.zeros(len(bin_edges) - 1))
        assert len(initial_varied_hists) == 2, len(initial_varied_hists)
        assert len(self._sys_weight) == len(data), (len(self._sys_weight), len(data))
        wc = weights > 0.
        weights_up = copy.copy(weights)
        weights_up[wc] = weights[wc] / self._sys_weight[wc] * (self._sys_weight[wc] + self._sys_uncert[wc])
        weights_dw = copy.copy(weights)
        weights_dw[wc] = weights[wc] / self._sys_weight[wc] * (self._sys_weight[wc] - self._sys_uncert[wc])

        bins = [np.array(list(edges)) for edges in bin_edges] if bin_edges is not None else bin_edges
        hist_up, _ = np.histogramdd(data, bins=bins, weights=weights_up)
        hist_dw, _ = np.histogramdd(data, bins=bins, weights=weights_dw)

        return initial_varied_hists[0] + hist_up, initial_varied_hists[1] + hist_dw

    @staticmethod
    def get_cov_from_varied_hists(varied_hists: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        assert len(varied_hists) == 2, len(varied_hists)
        hist_up, hist_dw = varied_hists
        diff_sym = (hist_up - hist_dw) / 2.
        return np.outer(diff_sym, diff_sym)


class SystematicsInfoItemFromVariation(SystematicsInfoItem):
    def __init__(self, sys_weight: np.ndarray, sys_uncert: np.ndarray):
        super().__init__()
        assert isinstance(sys_uncert, np.ndarray), type(sys_uncert)
        assert len(sys_uncert.shape) == 2, sys_uncert.shape
        assert sys_uncert.shape[1] > 1, sys_uncert.shape
        assert len(sys_weight) == len(sys_uncert), (sys_weight.shape, sys_uncert.shape)

        self._sys_type = "variation"
        self._sys_weight = sys_weight
        self._sys_uncert = sys_uncert

    def number_of_variations(self):
        return self._sys_uncert.shape[1]

    def get_cov(
            self,
            data: Optional[np.ndarray] = None,
            weights: WeightsInputType = None,
            bin_edges: Optional[BinEdgesType] = None
    ) -> np.ndarray:
        varied_hists = self.get_varied_hist(initial_varied_hists=None, data=data, weights=weights, bin_edges=bin_edges)
        return self.get_cov_from_varied_hists(varied_hists=varied_hists)

    def get_varied_hist(
            self,
            initial_varied_hists: Optional[Tuple[np.ndarray, ...]],
            data: Optional[np.ndarray] = None,
            weights: WeightsInputType = None,
            bin_edges: Optional[BinEdgesType] = None
    ) -> Tuple[np.ndarray, ...]:
        assert data is not None
        assert weights is not None
        assert bin_edges is not None
        if initial_varied_hists is None:
            initial_varied_hists = (np.zeros(len(bin_edges) - 1),) * self.number_of_variations()
        assert len(initial_varied_hists) == self.number_of_variations(), (len(initial_varied_hists),
                                                                          self.number_of_variations())
        assert len(self._sys_weight) == len(data), (len(self._sys_weight), len(data))

        varied_hists = []
        for hist_variation, sys_weight_var in zip(initial_varied_hists, self._sys_uncert.T):
            varied_weights = copy.copy(weights)
            w_cond = weights > 0.
            varied_weights[w_cond] = weights[w_cond] / self._sys_weight[w_cond] * sys_weight_var[w_cond]

            bins = [np.array(list(edges)) for edges in bin_edges] if bin_edges is not None else bin_edges
            varied_hists.append(hist_variation + np.histogramdd(data, bins=bins, weights=varied_weights)[0])

        assert len(varied_hists) == len(initial_varied_hists), (len(varied_hists), len(initial_varied_hists))
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
            weights: WeightsInputType = None
    ):
        self._sys_info_list = self._get_sys_info(in_systematics=in_sys, data=data, in_data=in_data, weights=weights)
        super().__init__()

    def _get_sys_info(
            self,
            in_systematics: SystematicsInputType,
            data: np.ndarray,
            in_data: Optional[pd.DataFrame],
            weights: WeightsInputType
    ) -> List[Union[None, SystematicsInfoItem]]:
        if in_systematics is None:
            return []

        # If not None, systematics must be provided as Tuple for one or List of Tuples for multiple.
        if isinstance(in_systematics, np.ndarray):
            return [self._get_sys_info_from_cov_matrix(in_systematics)]
        elif isinstance(in_systematics, tuple):
            return [self._get_single_sys_info(in_systematics, data, in_data, weights)]
        elif isinstance(in_systematics, list):
            return self._get_sys_info_from_list(in_systematics, data, in_data, weights)
        else:
            raise ValueError(f"Provided systematics has unexpected type {type(in_systematics)}.")

    @staticmethod
    def _get_sys_info_from_cov_matrix(in_systematics: SystematicsInputType) -> SystematicsInfoItem:
        assert isinstance(in_systematics, np.ndarray), type(in_systematics)
        assert len(in_systematics.shape) == 2, len(in_systematics.shape)
        assert in_systematics.shape[0] == in_systematics.shape[1], (in_systematics.shape[0], in_systematics.shape[1])
        return SystematicsInfoItemFromCov(cov_matrix=in_systematics)

    def _get_single_sys_info(
            self,
            in_systematics: SystematicsInputType,
            data: np.ndarray,
            in_data: Optional[pd.DataFrame],
            weights: WeightsInputType
    ) -> Union[None, SystematicsInfoItem]:
        if in_systematics is None:
            return None
        if len(in_systematics) == 1:
            return self._get_sys_info_from_cov_matrix(in_systematics)
        elif len(in_systematics) == 2:
            sys_weight = Weights.obtain_weights(weight_input=in_systematics[0], data=data, data_input=in_data)
            assert len(sys_weight) == len(data), (len(sys_weight), len(data))
            assert len(sys_weight) == len(weights)
            assert not np.isnan(sys_weight).any()
            assert np.all(sys_weight[weights > 0.] > 0.)

            if isinstance(in_systematics[1], list):
                variations = [Weights.obtain_weights(s, data, in_data) for s in in_systematics[1]]
                sys_uncert = np.column_stack((variation for variation in variations))
                assert sys_uncert.shape[1] == len(in_systematics[1]), (sys_uncert.shape, len(in_systematics[1]))
                assert not np.isnan(sys_uncert).any()
                return SystematicsInfoItemFromVariation(sys_weight=sys_weight, sys_uncert=sys_uncert)
            else:
                sys_uncert = Weights.obtain_weights(weight_input=in_systematics[1], data=data, data_input=in_data)
                assert not np.isnan(sys_uncert).any()
                return SystematicsInfoItemFromUpDown(sys_weight=sys_weight, sys_uncert=sys_uncert)
        else:
            raise ValueError(f"Systematics must be provided as tuple or list of tuples"
                             f"or directly as the respective covariance matrix. "
                             f"Each tuple must contain 2 entries!\n"
                             f"A provided tuple was of size {len(in_systematics)} != 2.")

    def _get_sys_info_from_list(
            self,
            in_systematics: SystematicsInputType,
            data: np.ndarray,
            in_data: Optional[pd.DataFrame],
            weights: WeightsInputType
    ) -> List[Union[None, SystematicsInfoItem]]:
        if len(in_systematics) == 0:
            return []

        result = [self._get_single_sys_info(in_sys, data, in_data, weights) for in_sys in in_systematics]
        return [e for e in result if e is not None]

    @property
    def as_list(self) -> List[Union[None, SystematicsInfoItem]]:
        return self._sys_info_list

    def __getitem__(self, i) -> Optional[SystematicsInfoItem]:
        return self._sys_info_list[i]

    def __len__(self) -> int:
        return len(self._sys_info_list)
