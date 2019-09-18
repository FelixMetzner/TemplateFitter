# TODO: Docstring
#  class which handles computation

import numpy as np
from numba import jit
from scipy.linalg import block_diag
from abc import ABC, abstractmethod

from templatefitter.utility import xlogyx  # TODO: Check if this has been changed!
from templatefitter.model.parameter_handler import ParameterHandler

__all__ = ["ModelBuilder"]


class ModelBuilder:
    def __init__(self, params, data):
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
        self.has_data = 1
        self.shape = ()
        self.converter_matrix = None
        self.converter_vector = None
        self.num_fractions = 0
        self.num_templates = 0
        self.num_bins = None

    def add_template(self, template, value, create=True, same=True):
        if self.num_bins is None:
            self.num_bins = template.num_bins
        self.packed_templates[template.name] = template
        if template._num_templates > 1:
            self.add_multi_template(template, value, create, same)
        else:
            self.templates[template.name] = template
            if create:
                yield_index = self.params.add_parameter("{}_yield".format(template.name), value)
                self.yield_indices.append(yield_index)
            else:
                self.yield_indices.append(value)

        if self._dim is None:
            self._dim = len(template.bins)
        self.num_templates = len(self.templates)

    def add_multi_template(self, template, value, create=True, same=True):
        self.subfraction_indices += template._par_indices
        self.num_fractions += len(template._par_indices)
        if create:
            if same:
                yield_index = self.params.add_parameter(
                    "{}_yield".format(template.name), value)
            for sub_temp in template._templates.values():
                self.templates[sub_temp.name] = sub_temp
                if not same:
                    yield_index = self.params.add_parameter(
                        "{}_yield".format(sub_temp.name), value[sub_temp.name])
                self.yield_indices.append(yield_index)
        else:
            for sub_temp in template._templates.values():
                self.templates[sub_temp.name] = sub_temp
                self.yield_indices.append(value[sub_temp.name])

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

        bin_mids = [template.bin_mids() for template in self.templates.values()]
        bin_edges = next(iter(self.templates.values())).bin_edges()
        bin_width = next(iter(self.templates.values())).bin_widths()
        num_bins = next(iter(self.templates.values())).num_bins
        shape = next(iter(self.templates.values())).shape()

        colors = [template.color for template in self.templates.values()]
        yields = self.params.get_parameters([self.yield_indices])
        bin_counts = [temp_yield * temp.fractions() for temp_yield, temp in zip(yields, self.templates.values())]
        labels = [template.name for template in self.templates.values()]

        if plot_all:
            colors = []
            for template in self.templates.values():
                colors += template.colors()
            labels = []
            for template in self.templates.values():
                labels += template.labels()
            bin_counts = [tempyield * template.all_fractions() for tempyield, template in
                          zip(yields, self.templates.values())]
            bin_counts = np.concatenate(bin_counts)
            N = len(bin_counts)
            bin_counts = np.split(bin_counts, N / num_bins)
            bin_mids = [bin_mids[0]] * int(N / num_bins)

        if self._dim > 1:
            bin_counts = [self._get_projection(kwargs["projection"], bc.reshape(shape)) for bc in bin_counts]
            axis = kwargs["projection"]
            ax_to_index = {
                "x": 0,
                "y": 1,
            }
            bin_mids = [mids[ax_to_index[axis]] for mids in bin_mids]
            bin_edges = bin_edges[ax_to_index[axis]]
            bin_width = bin_width[ax_to_index[axis]]

        ax.hist(
            bin_mids,
            weights=bin_counts,
            bins=bin_edges,
            edgecolor="black",
            histtype="stepfilled",
            lw=0.5,
            color=colors,
            label=labels,
            stacked=True
        )

        uncertainties_sq = [
            (temp_yield * template.fractions() * template.errors()).reshape(template.shape()) ** 2
            for temp_yield, template in zip(yields, self.templates.values())
        ]
        if self._dim > 1:
            uncertainties_sq = [self._get_projection(kwargs["projection"], unc_sq) for unc_sq in uncertainties_sq]

        total_uncertainty = np.sqrt(np.sum(np.array(uncertainties_sq), axis=0))
        total_bin_count = np.sum(np.array(bin_counts), axis=0)

        ax.bar(
            x=bin_mids[0],
            height=2 * total_uncertainty,
            width=bin_width,
            bottom=total_bin_count - total_uncertainty,
            color="black",
            hatch="///////",
            fill=False,
            lw=0,
            label="MC Uncertainty"
        )

        if self.data is None:
            return ax

        data_bin_mids = self.data.bin_mids
        data_bin_counts = self.data.bin_counts
        data_bin_errors_sq = self.data.bin_errors_sq

        if self.has_data:

            if self._dim > 1:
                data_bin_counts = self._get_projection(
                    kwargs["projection"], data_bin_counts
                )
                data_bin_errors_sq = self._get_projection(
                    kwargs["projection"], data_bin_errors_sq
                )

                axis = kwargs["projection"]
                ax_to_index = {
                    "x": 0,
                    "y": 1,
                }
                data_bin_mids = data_bin_mids[ax_to_index[axis]]

            ax.errorbar(x=data_bin_mids, y=data_bin_counts, yerr=np.sqrt(data_bin_errors_sq),
                        ls="", marker=".", color="black", label="Data")

    def create_nll(self):
        return CostFunction(self, self.params)


class AbstractTemplateCostFunction(ABC):
    """
    Abstract base class for all cost function to estimate
    yields using the template method.
    """

    def __init__(self):
        pass

    # -- abstract properties

    @property
    @abstractmethod
    def x0(self):
        """
        numpy.ndarray: Starting values for the minimization.
        """
        pass

    @property
    @abstractmethod
    def param_names(self):
        """
        list of str: Parameter names. Used for convenience.
        """
        pass

    # -- abstract methods --

    @abstractmethod
    def __call__(self, x: np.ndarray) -> float:
        pass


class CostFunction(AbstractTemplateCostFunction):

    def __init__(self, model: ModelBuilder, params: ParameterHandler):
        super().__init__()
        self._model = model
        self._params = params

    @property
    def x0(self):
        return self._params.get_parameters()

    @property
    def param_names(self):
        return self._params.get_parameter_names()

    def __call__(self, x):
        return self._model.chi2(x)
