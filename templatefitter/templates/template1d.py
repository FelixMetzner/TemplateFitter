import logging
import numpy as np

from templatefitter.histograms import Hist1d
from templatefitter.templates import SingleTemplate

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["Template1D"]


class Template1D(SingleTemplate):

    def __init__(
            self,
            name,
            variable,
            hist1d,
            params,
            color=None,
            pretty_variable=None,
            pretty_label=None,
    ):
        super().__init__(name=name, params=params)

        self._hist = hist1d
        self._flat_bin_counts = self._hist.bin_counts.flatten()
        self._flat_bin_errors_sq = self._hist.bin_errors_sq.flatten()
        self._bins = hist1d.shape
        self._num_bins = hist1d.num_bins
        self._range = hist1d.range
        self._cov_mats = list()
        self._cov = None
        self._corr = None
        self._inv_corr = None
        self._relative_errors = None
        self._bin_par_indices = []
        self._fraction_function = self.bin_fractions
        self._sys_errors = []
        self._sys_par_indices = []
        self._n_up_vars = np.array([])
        self._n_down_vars = np.array([])
        self._up_vars = []
        self._down_vars = []
        # self._init_params()

        self._init_errors()

        self._variable = variable
        self.pretty_variable = pretty_variable
        self.color = color
        self.pretty_label = pretty_label

    def _init_params(self):
        bin_pars = np.full(self._num_bins, 0.)
        bin_par_names = ["{}_binpar_{}".format(self.name, i) for i in range(0, self._num_bins)]
        self._bin_par_indices = self._params.addParameters(bin_pars, bin_par_names)

    def colors(self):
        return [self.color]

    def labels(self):
        return [self.name]

    def add_variation(self, data, weights_up, weights_down):
        """
        Add a new covariance matrix from a given systematic variation
        of the underlying histogram to the template.
        """
        hup = Hist1d(bins=self._hist.num_bins, hist_range=self._range, data=data, weights=weights_up)
        hdown = Hist1d(bins=self._hist.num_bins, hist_range=self._range, data=data, weights=weights_down)
        self._add_cov_mat(hup, hdown)

    def add_single_par_variation(self, data, weights_up, weights_down, name, register=True):
        """
        Add a new covariance matrix from a given systematic variation
        of the underlying histogram to the template.
        """
        hup = Hist1d(bins=self._hist.num_bins, hist_range=self._range, data=data, weights=weights_up)
        hdown = Hist1d(bins=self._hist.num_bins, hist_range=self._range, data=data, weights=weights_down)

        self._up_vars.append(list(hup.bin_counts.flatten() - self._flat_bin_counts))
        self._down_vars.append(list(hdown.bin_counts.flatten() - self._flat_bin_counts))
        self._n_up_vars = np.array(self._up_vars)
        self._n_down_vars = np.array(self._down_vars)
        if register:
            self._params.addParameter(name, 0.)
        else:
            self._sys_par_indices.append(self._params.getIndex(name))
        self._fraction_function = self.bin_fractions_with_sys
