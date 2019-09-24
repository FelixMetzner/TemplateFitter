"""
This class provides a general Systematics class
"""

import copy
import numpy as np

from typing import Optional
from abc import ABC, abstractmethod
from collections.abc import Sequence

from templatefitter.binned_distributions.weights import Weights


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


class SystematicsInfoItemFromCov(SystematicsInfoItem):
    def __init__(self, cov_matrix: np.ndarray):
        super().__init__()
        assert isinstance(cov_matrix, np.ndarray), type(cov_matrix)
        assert len(cov_matrix.shape) == 2, cov_matrix.shape
        assert cov_matrix.shape[0] == cov_matrix.shape[1], cov_matrix.shape
        self._sys_type = "cov_matrix"
        self._cov_matrix = cov_matrix

    def get_cov(self, data=None, weights=None, bin_edges=None):
        assert len(bin_edges) - 1 == self._cov_matrix.shape[0], (len(bin_edges) - 1, self._cov_matrix.shape[0])
        assert len(bin_edges) - 1 == self._cov_matrix.shape[1], (len(bin_edges) - 1, self._cov_matrix.shape[1])
        return self._cov_matrix

    def get_varied_hist(self, initial_varied_hists, data=None, weights=None, bin_edges=None):
        raise NotImplementedError("This method is not (yet) supported for systematics provided via covariance matrix.")

    @staticmethod
    def get_cov_from_varied_hists(varied_hists):
        raise NotImplementedError("This method is not (yet) supported for systematics provided via covariance matrix.")


class SystematicsInfoItemFromUpDown(SystematicsInfoItem):
    def __init__(self, sys_weight, sys_uncert):
        super().__init__()
        self._sys_type = "up_down"
        assert isinstance(sys_uncert, np.ndarray), type(sys_uncert)
        assert len(sys_uncert.shape) == 1, sys_uncert.shape
        assert len(sys_weight) == len(sys_uncert), (sys_weight.shape, sys_uncert.shape)
        self._sys_weight = sys_weight
        self._sys_uncert = sys_uncert

    def get_cov(self, data=None, weights=None, bin_edges=None):
        varied_hists = self.get_varied_hist(initial_varied_hists=None, data=data, weights=weights, bin_edges=bin_edges)
        return self.get_cov_from_varied_hists(varied_hists=varied_hists)

    def get_varied_hist(self, initial_varied_hists, data=None, weights=None, bin_edges=None):
        if initial_varied_hists is None:
            initial_varied_hists = (np.zeros(len(bin_edges) - 1), np.zeros(len(bin_edges) - 1))
        assert len(initial_varied_hists) == 2, len(initial_varied_hists)
        assert len(self._sys_weight) == len(data), (len(self._sys_weight), len(data))
        wc = weights > 0.
        weights_up = copy.copy(weights)
        weights_up[wc] = weights[wc] / self._sys_weight[wc] * (self._sys_weight[wc] + self._sys_uncert[wc])
        weights_dw = copy.copy(weights)
        weights_dw[wc] = weights[wc] / self._sys_weight[wc] * (self._sys_weight[wc] - self._sys_uncert[wc])
        hist_up = np.histogram(data, bins=bin_edges, weights=weights_up)[0]
        hist_dw = np.histogram(data, bins=bin_edges, weights=weights_dw)[0]

        return initial_varied_hists[0] + hist_up, initial_varied_hists[1] + hist_dw

    @staticmethod
    def get_cov_from_varied_hists(varied_hists):
        assert len(varied_hists) == 2, len(varied_hists)
        hist_up, hist_dw = varied_hists
        diff_sym = (hist_up - hist_dw) / 2.
        return np.outer(diff_sym, diff_sym)


class SystematicsInfoItemFromVariation(SystematicsInfoItem):
    def __init__(self, sys_weight, sys_uncert):
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

    def get_cov(self, data=None, weights=None, bin_edges=None):
        varied_hists = self.get_varied_hist(initial_varied_hists=None, data=data, weights=weights, bin_edges=bin_edges)
        return self.get_cov_from_varied_hists(varied_hists=varied_hists)

    def get_varied_hist(self, initial_varied_hists, data=None, weights=None, bin_edges=None):
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
            varied_hists.append(hist_variation + np.histogram(data, bins=bin_edges, weights=varied_weights)[0])

        assert len(varied_hists) == len(initial_varied_hists), (len(varied_hists), len(initial_varied_hists))
        return tuple(varied_hists)

    @staticmethod
    def get_cov_from_varied_hists(varied_hists):
        cov = np.cov(np.column_stack(varied_hists))
        assert cov.shape[0] == cov.shape[1] == len(varied_hists[0]), (cov.shape[0], cov.shape[1], len(varied_hists[0]))
        assert not np.isnan(cov).any()
        return cov


class SystematicsInfo(Sequence):
    def __init__(
            self,
            in_sys=None,
            data=None,
            in_data=None,
            weights=None
    ):
        self._sys_info_list = self._get_sys_info(in_systematics=in_sys, data=data, in_data=in_data, weights=weights)
        super().__init__()

    def _get_sys_info(self, in_systematics, data, in_data, weights):
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
    def _get_sys_info_from_cov_matrix(in_systematics):
        assert isinstance(in_systematics, np.ndarray), type(in_systematics)
        assert len(in_systematics.shape) == 2, len(in_systematics.shape)
        assert in_systematics.shape[0] == in_systematics.shape[1], (in_systematics.shape[0], in_systematics.shape[1])
        return SystematicsInfoItemFromCov(cov_matrix=in_systematics)

    def _get_single_sys_info(self, in_systematics, data, in_data, weights):
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

    def _get_sys_info_from_list(self, in_systematics, data, in_data, weights):
        if len(in_systematics) == 0:
            return []

        result = [self._get_single_sys_info(in_sys, data, in_data, weights) for in_sys in in_systematics]
        return [e for e in result if e is not None]

    @property
    def as_list(self):
        return self._sys_info_list

    def __getitem__(self, i):
        return self._sys_info_list[i]

    def __len__(self):
        return len(self._sys_info_list)


def _get_cov_from_systematics(self, component_label: Optional[str] = None) -> Optional[np.ndarray]:
    if component_label is not None:
        assert component_label in [c.label for c in self._mc_components["single"]]
        components = [c for c in self._mc_components["single"] if c.label == component_label]
        assert len(components) == 1
        comp = components[0]
        if comp.systematics is None:
            return None

        cov = np.zeros((len(self._bin_mids), len(self._bin_mids)))
        for sys_info in comp.systematics:
            cov += sys_info.get_cov(data=comp.data, weights=comp.weights, bin_edges=self.bin_edges())
        return cov
    else:
        components = self._mc_components["stacked"]
        if all(comp.systematics is None for comp in components):
            return None
        if all(len(comp.systematics) == 0 for comp in components):
            return None

        assert all(len(comp.systematics) == len(components[0].systematics) for comp in components)

        cov = np.zeros((len(self._bin_mids), len(self._bin_mids)))
        for sys_index in range(len(components[0].systematics)):
            assert all(isinstance(comp.systematics[sys_index], type(components[0].systematics[sys_index]))
                       for comp in components)

            varied_hists = None
            for comp in components:
                varied_hists = comp.systematics[sys_index].get_varied_hist(
                    initial_varied_hists=varied_hists,
                    data=comp.data,
                    weights=comp.weights,
                    bin_edges=self.bin_edges
                )

            cov += components[0].systematics[sys_index].get_cov_from_varied_hists(varied_hists=varied_hists)

        return cov
