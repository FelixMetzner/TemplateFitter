# class containing core functionality for 2D and 1D templates

import logging
import numpy as np

from abc import ABC, abstractmethod

from templatefitter.utility import cov2corr, get_systematic_cov_mat
from templatefitter.templates.abstract_template import AbstractTemplate

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["SingleTemplate"]


class SingleTemplate(AbstractTemplate, ABC):
    """
    Defines the template interface.
    """

    def __init__(self, name, params):
        super(SingleTemplate, self).__init__(name=name, params=params)
        self._hist = None
        self._flat_bin_counts = None
        self._flat_bin_errors_sq = None
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
        self._num_templates = 1

    def shape(self):
        """tuple of int: Template shape."""
        return self._hist.shape

    def bin_mids(self):
        return self._hist.bin_mids

    def bin_edges(self):
        return self._hist.bin_edges

    def bin_widths(self):
        return self._hist.bin_widths

    def fractions(self):
        """
        Calculates the per bin fraction :math:`f_i` of the template.
        This value is used to calculate the expected number of events
        per bin :math:`\\nu_i` as :math:`\\nu_i=f_i\cdot\\nu`, where
        :math:`\\nu` is the expected yield. The fractions are given as

        .. math::

            f_i=\sum\limits_{i=1}^{n_\mathrm{bins}} \\frac{\\nu_i(1+\\theta_i\cdot\epsilon_i)}{\sum_{j=1}^{n_\mathrm{bins}} \\nu_j (1+\\theta_j\cdot\epsilon_j)},

        where :math:`\\theta_j` are the nuissance parameters and
        :math:`\epsilon_j` are the relative uncertainties per bin.

        Parameters
        ----------
        nui_params : numpy.ndarray
            An array with values for the nuissance parameters.
            Shape is (`num_bins`,)

        Returns
        -------
        numpy.ndarray
            Bin fractions of this template. Shape is (`num_bins`,).
        """
        return self._fraction_function()

    def all_fractions(self):
        return self.fractions()

    def bin_fractions(self):
        per_bin_yields = self._flat_bin_counts * (
                1. + self._params.getParameters([self._bin_par_indices]) * self._relative_errors)
        return per_bin_yields / np.sum(per_bin_yields)

    def bin_fractions_with_sys(self):
        per_bin_yields = self._flat_bin_counts * (
                1. + self._params.getParameters([self._bin_par_indices]) * self._relative_errors)
        sys_pars = self._params.getParameters([self._sys_par_indices])[:, np.newaxis]
        uperrs = sys_pars * (sys_pars > 0) * self._up_vars
        downerrs = sys_pars * (sys_pars < 0) * self._down_vars
        sys_corr = np.product(uperrs + downerrs + 1, axis=0)
        per_bin_yields = per_bin_yields * sys_corr
        return per_bin_yields / np.sum(per_bin_yields)

    def _init_errors(self):
        """
        The statistical covariance matrix is initialized as diagonal
        matrix of the sum of weights squared per bin in the underlying
        histogram. For empty bins, the error is set to 1e-7. The errors
        are initialized to be 100% uncorrelated. The relative errors per
        bin are set to 1e-7 in case of empty bins.
        """
        stat_errors_sq = np.copy(self._flat_bin_errors_sq)
        stat_errors_sq[stat_errors_sq == 0] = 1e-14

        self._cov = np.diag(stat_errors_sq)

        self._cov_mats.append(np.copy(self._cov))

        self._corr = np.diag(np.ones(self._num_bins))
        self._inv_corr = np.diag(np.ones(self._num_bins))

        self._relative_errors = np.divide(
            np.sqrt(np.diag(self._cov)),
            self._flat_bin_counts,
            out=np.full(self._num_bins, 1e-7),
            where=self._flat_bin_counts != 0,
        )

    def _add_cov_mat(self, hup, hdown):
        """
        Helper function. Calculates a covariance matrix from
        given histogram up and down variations.
        """
        cov_mat = get_systematic_cov_mat(self._flat_bin_counts, hup.bin_counts.flatten(), hdown.bin_counts.flatten())
        self._cov_mats.append(np.copy(cov_mat))

        self._cov += cov_mat
        self._corr = cov2corr(self._cov)
        self._inv_corr = np.linalg.inv(self._corr)
        self._relative_errors = np.divide(
            np.sqrt(np.diag(self._cov)),
            self._flat_bin_counts,
            out=np.full(self._num_bins, 1e-7),
            where=self._flat_bin_counts != 0,
        )

    @property
    def values(self, index):
        """
        Calculates the expected number of events per bin using
        the current yield value and nuissance parameters. Shape
        is (`num_bins`,).
        """
        yield_param = self.params.getParameter(index)
        return yield_param * self.fractions().reshape(self._hist.shape)

    def reshaped_fractions(self):
        """
        Calculates the expected number of events per bin using
        the current yield value and nuissance parameters. Shape
        is (`num_bins`,).
        """
        return self.fractions().reshape(self._hist.shape)

    def errors(self):
        """
        numpy.ndarray: Total uncertainty per bin. This value is the
        product of the relative uncertainty per bin and the current bin
        values. Shape is (`num_bins`,).
        """
        return self._relative_errors

    def reshaped_errors(self):
        """
        numpy.ndarray: Total uncertainty per bin. This value is the
        product of the relative uncertainty per bin and the current bin
        values. Shape is (`num_bins`,).
        """
        return self._relative_errors.reshape(self._hist.shape)

    @property
    def cov_mat(self):
        return self._cov

    @property
    def cov_mats(self):
        return self._cov_mats

    @property
    def corr(self):
        return self._corr

    def inv_corr_mat(self):
        return self._inv_corr

    @abstractmethod
    def add_variation(self, data, weights_up, weights_down):
        pass

    @abstractmethod
    def add_single_par_variation(self, data, weights_up, weights_down):
        pass

    def get_bin_pars(self):
        return self._params.getParameters(self._bin_par_indices)
