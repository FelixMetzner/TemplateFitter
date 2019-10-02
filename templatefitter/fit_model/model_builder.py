"""
Class which defines the fit model by combining templates and which handles the computation.
"""

import logging
import numpy as np
from numba import jit
from scipy.linalg import block_diag
from abc import ABC, abstractmethod
from typing import Union, Dict, List

from templatefitter.utility import xlogyx
from templatefitter.fit_model.parameter_handler import ParameterHandler
from templatefitter.plotter import old_plotting

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["ModelBuilder"]


class ModelBuilder:
    def __init__(
            self,
            parameter_handler: ParameterHandler,
            data  # TODO: Type hint
    ):
        self._data = data
        self._params = parameter_handler

        # TODO:
        self.templates = {}
        self.packed_templates = {}
        self.data = data
        self.x_obs = data.bin_counts.flatten()
        self.x_obs_errors = data.bin_errors.flatten()
        self.yield_indices = []
        self.subfraction_indices = []
        self.constrain_indices = []
        self.constrain_value = np.array([])
        self.constrain_sigma = np.array([])
        self._inv_corr = np.array([])
        self.bin_par_slice = (0, 0)
        self._dim = None
        self.has_data = True
        self.shape = ()
        self.converter_matrix = None
        self.converter_vector = None
        self.num_fractions = 0
        self.num_templates = 0
        self.num_bins = None

    # TODO: Check that every template of a model uses the same ParameterHandler instance!
    # TODO: Possible Check: For first call of expected_events_per_bin: Check if template indices are ordered correctly.

    def setup_model_from_channel_container(self):
        # TODO: Keep track of binning of all channels!
        # TODO: Keep track of number of channels!
        # TODO: Keep track of number of components per channel
        # TODO: Keep track of number of number of templates

        # TODO: keep track of fractions for each channel...
        #         self.subfraction_indices += template._par_indices
        #         self.num_fractions += len(template._par_indices)

        pass

    def template_matrix(self):
        """ Creates the fixed template stack """
        fractions_per_template = [template._flat_bin_counts for template in self.templates.values()]

        self.template_fractions = np.stack(fractions_per_template)
        self.shape = self.template_fractions.shape

    def relative_error_matrix(self):
        errors_per_template = [template.errors() for template
                               in self.templates.values()]

        self.template_errors = np.stack(errors_per_template)

    def initialise_bin_pars(self):
        """ Add parameters for the template """

        bin_pars = np.zeros((self.num_bins * len(self.templates), 1))
        bin_par_names = []
        for template in self.templates.values():
            bin_par_names += ["{}_binpar_{}".format(template.name, i) for i in range(0, self.num_bins)]
        bin_par_indices = self._params.add_parameters(bin_pars, bin_par_names)
        self.bin_par_slice = (bin_par_indices[0], bin_par_indices[-1] + 1)

    @jit
    def expected_events_per_bin(self, bin_pars: np.ndarray, yields: np.ndarray, sub_pars: np.ndarray) -> np.ndarray:
        sys_pars = self._params.get_parameters_by_index(self.sys_pars)
        # compute the up and down errors for single par variations
        up_corr = np.prod(1 + sys_pars * (sys_pars > 0) * self.up_errors, 0)
        down_corr = np.prod(1 + sys_pars * (sys_pars < 0) * self.down_errors, 0)
        corrections = (1 + self.template_errors * bin_pars) * (up_corr + down_corr)
        sub_fractions = np.matmul(self.converter_matrix, sub_pars) + self.converter_vector
        fractions = self.template_fractions * corrections
        norm_fractions = fractions / np.sum(fractions, 1)[:, np.newaxis]
        expected_per_bin = np.sum(norm_fractions * yields * sub_fractions, axis=0)
        return expected_per_bin
        # compute overall correction terms
        # get sub template fractions into the correct form with the converter and additive part
        # normalised expected corrected fractions
        # determine expected amount of events in each bin

    def fraction_converter(self) -> None:
        """
        Determines the matrices required to transform the sub-template parameters
        """
        arrays = []
        additive = []
        count = 0
        for template in self.packed_templates.values():
            if template._num_templates == 1:
                arrays.append(np.zeros((1, self.num_fractions)))
                additive.append(np.ones((1, 1)))
            else:
                n_fractions = template._num_templates - 1
                array = np.identity(n_fractions)
                array = np.vstack([array, np.full((1, n_fractions), -1.)])
                count += n_fractions
                array = np.pad(array, ((0, 0), (count - n_fractions, self.num_fractions - count)), mode='constant')
                arrays.append(array)
                additive.append(np.vstack([np.zeros((n_fractions, 1)), np.ones((1, 1))]))
        print(arrays)
        print(additive)
        self.converter_matrix = np.vstack(arrays)
        self.converter_vector = np.vstack(additive)

    def add_constraint(self, name: str, value: float, sigma: float) -> None:
        self.constrain_indices.append(self._params.get_index(name))
        self.constrain_value = np.append(self.constrain_value, value)
        self.constrain_sigma = np.append(self.constrain_sigma, sigma)

    def x_expected(self) -> np.ndarray:
        yields = self._params.get_parameters_by_index(self.yield_indices)
        fractions_per_template = np.array([template.fractions() for template in self.templates.values()])
        return yields @ fractions_per_template

    def bin_pars(self) -> np.ndarray:
        return np.concatenate([template.get_bin_pars() for template in self.templates.values()])

    def _create_block_diag_inv_corr_mat(self) -> None:
        inv_corr_mats = [template.inv_corr_mat() for template in self.templates.values()]
        self._inv_corr = block_diag(*inv_corr_mats)

    def _constrain_term(self) -> float:
        constrain_pars = self._params.get_parameters_by_index(self.constrain_indices)
        chi2constrain = np.sum(((self.constrain_value - constrain_pars) / self.constrain_sigma) ** 2)
        assert isinstance(chi2constrain, float), type(chi2constrain)  # TODO: Remove this assertion for speed-up!
        return chi2constrain

    @jit
    def _gauss_term(self, bin_pars: np.ndarray) -> float:
        return bin_pars @ self._inv_corr @ bin_pars

    @jit
    def chi2(self, pars: np.ndarray) -> float:
        self._params.set_parameters(pars)

        yields = self._params.get_parameters_by_index(self.yield_indices).reshape(self.num_templates, 1)
        sub_pars = self._params.get_parameters_by_index(self.subfraction_indices).reshape(self.num_fractions, 1)
        bin_pars = self._params.get_parameters_by_slice(self.bin_par_slice)

        chi2 = self.chi2_compute(bin_pars, yields, sub_pars)
        return chi2

    @jit
    def chi2_compute(self, bin_pars: np.ndarray, yields: np.ndarray, sub_pars: np.ndarray) -> float:
        chi2data = np.sum(
            (self.expected_events_per_bin(bin_pars.reshape(self.shape), yields, sub_pars) - self.x_obs) ** 2
            / (2 * self.x_obs_errors ** 2)
        )

        assert isinstance(chi2data, float), type(chi2data)  # TODO: Remove this assertion for speed-up!

        chi2 = chi2data + self._gauss_term(bin_pars)  # + self._constrain_term()  # TODO: Check this
        return chi2

    def nll(self, pars: np.ndarray) -> float:
        self._params.set_parameters(pars)

        exp_evts_per_bin = self.x_expected()
        poisson_term = np.sum(exp_evts_per_bin - self.x_obs - xlogyx(self.x_obs, exp_evts_per_bin))
        assert isinstance(poisson_term, float), type(poisson_term)  # TODO: Remove this assertion for speed-up!

        nll = poisson_term + (self._gauss_term() + self._constrain_term()) / 2.
        return nll

    @staticmethod
    def _get_projection(ax: str, bc: np.ndarray) -> np.ndarray:
        # TODO: Is the mapping for x and y defined the wrong way around?
        x_to_i = {
            "x": 1,
            "y": 0
        }

        # TODO: use method provided by BinnedDistribution!
        return np.sum(bc, axis=x_to_i[ax])

    # TODO: Use histogram for plotting!
    def plot_stacked_on(self, ax, plot_all=False, **kwargs):
        plot_info = old_plotting.PlottingInfo(
            templates=self.templates,
            params=self._params,
            yield_indices=self.yield_indices,
            dimension=self._dim,
            projection_fct=self._get_projection,
            data=self.data,
            has_data=self.has_data
        )
        return old_plotting.plot_stacked_on(plot_info=plot_info, ax=ax, plot_all=plot_all, **kwargs)

    # TODO: Problematic; At the moment some sort of forward declaration is necessary for type hint...
    def create_nll(self) -> CostFunction:
        return CostFunction(self, parameter_handler=self._params)


# TODO: Maybe relocate cost functions into separate sub-package;
#  however: CostFunction depends on ModelBuilder and vice versa ...
class AbstractTemplateCostFunction(ABC):
    """
    Abstract base class for all cost function to estimate yields using the template method.
    """

    def __init__(self, model: ModelBuilder, parameter_handler: ParameterHandler) -> None:
        self._model = model
        self._params = parameter_handler

    def x0(self) -> np.ndarray:
        """ Returns initial parameters of the model """
        return self._params.get_parameters()

    def param_names(self) -> List[str]:
        return self._params.get_parameter_names()

    @abstractmethod
    def __call__(self, x: np.ndarray) -> float:
        raise NotImplementedError(f"{self.__class__.__name__} is an abstract base class.")


class CostFunction(AbstractTemplateCostFunction):
    def __init__(self, model: ModelBuilder, parameter_handler: ParameterHandler) -> None:
        super().__init__(model=model, parameter_handler=parameter_handler)

    def __call__(self, x) -> float:
        return self._model.chi2(x)
