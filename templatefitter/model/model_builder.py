"""
Class which defines the fit model by combining templates and which handles the computation.
"""

import numpy as np
from numba import jit
from scipy.linalg import block_diag
from abc import ABC, abstractmethod
from typing import Union, Dict, List, Callable

from templatefitter.utility import xlogyx
from templatefitter.model.parameter_handler import ParameterHandler
from templatefitter.plotter import old_plotting

__all__ = ["ModelBuilder"]


class ModelBuilder:
    def __init__(
            self,
            params: ParameterHandler,
            data  # TODO: Type hint
    ):
        self.params = params
        self.templates = {}
        self.packed_templates = {}
        self.data = data
        self.x_obs = data.bin_counts.flatten()
        self.x_obs_errors = data.bin_errors.flatten()
        self.yield_indices = []
        self.subfraction_indices = []
        self.con_indices = []
        self.con_value = np.array([])
        self.con_sigma = np.array([])
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

    def add_template(
            self,
            template,  # TODO: Type hint
            parameter_value: float,  # TODO: Maybe handle this differently
            create: bool = True,
            same: bool = True
    ) -> None:
        if self.num_bins is None:
            self.num_bins = template.num_bins
        self.packed_templates[template.name] = template
        if template._num_templates > 1:
            self.add_multi_template(template, parameter_value, create, same)
        else:
            self.templates[template.name] = template
            if create:
                yield_index = self.params.add_parameter(parameter_value, "{}_yield".format(template.name))
                self.yield_indices.append(yield_index)
            else:
                self.yield_indices.append(parameter_value)

        if self._dim is None:
            self._dim = len(template.bins)
        self.num_templates = len(self.templates)

    def add_multi_template(
            self,
            template,  # TODO: Type hint
            parameter_value: Union[float, Dict[str, float]],  # TODO: Maybe handle this differently
            create: bool = True,
            same: bool = True
    ) -> None:
        self.subfraction_indices += template._par_indices
        self.num_fractions += len(template._par_indices)
        if create:
            if same:
                yield_index = self.params.add_parameter(parameter_value, "{}_yield".format(template.name))
            for sub_temp in template._templates.values():
                self.templates[sub_temp.name] = sub_temp
                if not same:
                    yield_index = self.params.add_parameter(parameter_value[sub_temp.name], "{}_yield".format(sub_temp.name))
                self.yield_indices.append(yield_index)
        else:
            for sub_temp in template._templates.values():
                self.templates[sub_temp.name] = sub_temp
                self.yield_indices.append(parameter_value[sub_temp.name])

    def template_matrix(self):
        """ Creates the fixed template stack """
        fractions_per_template = [template._flat_bin_counts for template
                                  in self.templates.values()]

        self.template_fractions = np.stack(fractions_per_template)
        self.shape = self.template_fractions.shape

    def relative_error_matrix(self):
        """ Creates the fixed template stack """
        errors_per_template = [template.errors() for template
                               in self.templates.values()]

        self.template_errors = np.stack(errors_per_template)

    def initialise_bin_pars(self):
        """ Add parameters for the template """

        bin_pars = np.zeros((self.num_bins * len(self.templates), 1))
        bin_par_names = []
        for template in self.templates.values():
            bin_par_names += ["{}_binpar_{}".format(template.name, i) for i in range(0, self.num_bins)]
        bin_par_indices = self.params.add_parameters(bin_pars, bin_par_names)
        self.bin_par_slice = (bin_par_indices[0], bin_par_indices[-1] + 1)

    @jit
    def expected_events_per_bin(self, bin_pars, yields, sub_pars):
        sys_pars = self.params.getParamters(self.sys_pars)
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

    def fraction_converter(self):
        """
        Determines the matrices required to transform the sub-template parameters
        """
        arrays = []
        additive = []
        count = 0
        for template in self.packed_templates.values():
            if template._num_templates == 1:
                arrays += [np.zeros((1, self.num_fractions))]
                additive += [np.full((1, 1), 1.)]
            else:
                n_fractions = template._num_templates - 1
                a = np.identity(n_fractions)
                a = np.vstack([a, np.full((1, n_fractions), -1.)])
                count += n_fractions
                a = np.pad(a, ((0, 0), (count - n_fractions, self.num_fractions - count)), mode='constant')
                arrays += [a]
                additive += [np.vstack([np.zeros((n_fractions, 1)), np.full((1, 1), 1.)])]
        print(arrays)
        print(additive)
        self.converter_matrix = np.vstack(arrays)
        self.converter_vector = np.vstack(additive)

    def add_constraint(self, name, value, sigma):
        self.con_indices.append(self.params.get_index(name))
        self.con_value = np.append(self.con_value, value)
        self.con_sigma = np.append(self.con_sigma, sigma)

    def x_expected(self):
        yields = self.params.get_parameters([self.yield_indices])
        fractions_per_template = np.array(
            [template.fractions() for template
             in self.templates.values()]
        )
        return yields @ fractions_per_template

    def bin_pars(self):
        return np.concatenate([template.get_bin_pars() for template in self.templates.values()])

    def _create_block_diag_inv_corr_mat(self):
        inv_corr_mats = [template.inv_corr_mat() for template in self.templates.values()]
        self._inv_corr = block_diag(*inv_corr_mats)

    def _con_term(self):
        con_pars = self.params.get_parameters([self.con_indices])
        chi2cons = np.sum(((self.con_value - con_pars) / self.con_sigma) ** 2)
        return chi2cons

    @jit
    def _gauss_term(self, bin_pars):
        return bin_pars @ self._inv_corr @ bin_pars

    @jit
    def chi2(self, pars):
        self.params.set_parameters(pars)
        yields = self.params.get_parameters_by_index(self.yield_indices).reshape(self.num_templates, 1)
        sub_pars = self.params.get_parameters_by_index(self.subfraction_indices).reshape(self.num_fractions, 1)
        bin_pars = self.params.get_parameters_by_slice(self.bin_par_slice)
        chi2 = self.chi2_compute(bin_pars, yields, sub_pars)
        return chi2

    @jit
    def chi2_compute(self, bin_pars, yields, sub_pars):
        chi2data = np.sum(
            (self.expected_events_per_bin(bin_pars.reshape(self.shape), yields, sub_pars) - self.x_obs) ** 2
            / (2 * self.x_obs_errors ** 2)
        )
        chi2 = chi2data + self._gauss_term(bin_pars)  # + self._con_term()
        return chi2

    def nll(self, pars):
        self.params.set_parameters(pars)
        exp_evts_per_bin = self.x_expected()
        poisson_term = np.sum(exp_evts_per_bin - self.x_obs -
                              xlogyx(self.x_obs, exp_evts_per_bin))

        nll = poisson_term + (self._gauss_term() + self._con_term()) / 2.
        return nll

    @staticmethod
    def _get_projection(ax, bc):
        x_to_i = {
            "x": 1,
            "y": 0
        }

        return np.sum(bc, axis=x_to_i[ax])

    def plot_stacked_on(self, ax, plot_all=False, **kwargs):
        plot_info = old_plotting.PlottingInfo(
            templates=self.templates,
            params=self.params,
            yield_indices=self.yield_indices,
            dimension=self._dim,
            projection_fct=self._get_projection,
            data=self.data,
            has_data=self.has_data
        )
        return old_plotting.plot_stacked_on(plot_info=plot_info, ax=ax, plot_all=plot_all, **kwargs)

    def create_nll(self):
        return CostFunction(self, self.params)


# Maybe relocate cost functions into separate sub-package...
class AbstractTemplateCostFunction(ABC):
    """
    Abstract base class for all cost function to estimate yields using the template method.
    """
    def __init__(self, model: ModelBuilder, params: ParameterHandler) -> None:
        self._model = model
        self._params = params

    def x0(self) -> np.ndarray:
        """ Returns initial parameters of the model """
        return self._params.get_parameters()

    def param_names(self) -> List[str]:
        return self._params.get_parameter_names()

    @abstractmethod
    def __call__(self, x: np.ndarray) -> Callable:
        raise NotImplementedError(f"{self.__class__.__name__} is an abstract base class.")


class CostFunction(AbstractTemplateCostFunction):
    def __init__(self, model: ModelBuilder, params: ParameterHandler) -> None:
        super().__init__(model=model, params=params)

    def __call__(self, x) -> Callable:
        return self._model.chi2(x)
