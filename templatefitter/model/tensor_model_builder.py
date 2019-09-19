# class which handles computation

import numpy as np
import tensorflow as tf
from scipy.linalg import block_diag

from templatefitter.utility import xlogyx
from templatefitter.plotter import old_plotting
from templatefitter.model.parameter_handler import ParameterHandler

__all__ = ["TensorModelBuilder"]


class TensorModelBuilder:
    def __init__(self, params, data, channels='False'):
        self.params = params
        self.templates = {}
        self.packed_templates = {}
        self.data = data
        self.x_obs = tf.constant(data.bin_counts.flatten(), dtype=tf.float64)
        self.x_obs_errors = tf.constant(data.bin_errors.flatten(), dtype=tf.float64)
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
        self.yields_to_stack = []
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
                self.yields_to_stack += [tf.Variable(value, dtype=tf.float64)]
            else:
                self.yields_to_stack += [value]

        if self._dim is None:
            self._dim = len(template.bins)
        self.num_templates = len(self.templates)

    def add_multi_template(self, template, value, create=True, same=True):
        self.subfraction_indices += template._par_indices
        self.num_fractions += len(template._par_indices)
        if create:
            if same:
                yield_index = tf.Variable(value, dtype=tf.float64)
            for sub_temp in template._templates.values():
                self.templates[sub_temp.name] = sub_temp
                if not same:
                    yield_index = tf.Variable(value[sub_temp.name], dtype=tf.float64)
                self.yields_to_stack += [yield_index]
        else:
            for sub_temp in template._templates.values():
                self.templates[sub_temp.name] = sub_temp
                self.yields_to_stack += [value[sub_temp.name]]

    def template_matrix(self):
        """ Creates the fixed template stack """
        fractions_per_template = [template._flat_bin_counts for template
                                  in self.templates.values()]

        self.template_fractions = np.stack(fractions_per_template)
        self.template_fractions = tf.constant(self.template_fractions, dtype=tf.float64)
        self.shape = self.template_fractions.shape

    def relative_error_matrix(self):
        """ Creates the fixed template stack """
        errors_per_template = [template.errors() for template
                               in self.templates.values()]

        self.template_errors = np.stack(errors_per_template)
        self.template_errors = tf.constant(self.template_errors, dtype=tf.float64)

    def initialise_tensor_variables(self):
        """ Add parameters for the template """
        self.bin_pars = tf.Variable(tf.zeros(self.shape, dtype=tf.float64))
        self.yields = tf.reshape(tf.stack(self.yields_to_stack), (self.num_templates, 1))
        sub_pars = self.params.get_parameters_by_index(self.subfraction_indices).reshape(self.num_fractions, 1)
        self.sub_pars = tf.Variable(sub_pars, dtype=tf.float64)

    def expected_events_per_bin(self):
        # Obtain required parameters
        # sys_pars = self.params.get_parameters(self.sys_pars)
        # compute the up and down errors for single par variations
        # up_corr = np.prod(1+sys_pars*(sys_pars>0)*self.up_errors,0)
        # down_corr = np.prod(1+sys_pars*(sys_pars<0)*self.down_errors,0)
        # compute overall correction terms
        corrections = self.template_errors * self.bin_pars + 1.
        # *(up_corr + down_corr)
        # get sub template fractions into the correct form with the converter and additive part
        # sub_fractions = tf.linalg.matmul(self.converter_matrix,tf.math.sigmoid(self.sub_pars)) + self.converter_vector
        sub_fractions = tf.linalg.matmul(self.converter_matrix, self.sub_pars) + self.converter_vector
        # normalised expected corrected fractions
        # fractions = self.template_fractions*corrections

        fractions = self.template_fractions * corrections
        norm_fractions = fractions / tf.reduce_sum(fractions, 1, keepdims=True)
        # determine expected amount of events in each bin
        expected_per_bin = tf.reduce_sum(norm_fractions * self.yields * sub_fractions, axis=0)
        # expected_per_bin = tf.reduce_sum(norm_fractions*self.yields,axis=0)
        return expected_per_bin

    def fraction_converter(self):
        """ Determines the matrices required to transform the sub-template . """
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
        self.converter_matrix = np.vstack(arrays)
        self.converter_vector = np.vstack(additive)
        self.converter_matrix = tf.constant(self.converter_matrix, dtype=tf.float64)
        self.converter_vector = tf.constant(self.converter_vector, dtype=tf.float64)

    def add_constraint(self, name, value, sigma):
        self.con_indices.append(self.params.get_index(name))
        self.con_value = np.append(self.con_value, value)
        self.con_sigma = np.append(self.con_sigma, sigma)

    def x_expected(self):
        yields = self.params.get_parameters([self.yield_indices])
        fractions_per_template = np.array([template.fractions() for template in self.templates.values()])
        return yields @ fractions_per_template

    def get_bin_pars(self):
        bin_pars = np.concatenate(
            [template.get_bin_pars() for template
             in self.templates.values()]
        )
        return bin_pars

    def _create_block_diag_inv_corr_mat(self):
        inv_corr_mats = [template.inv_corr_mat() for template
                         in self.templates.values()]
        self._inv_corr = block_diag(*inv_corr_mats)
        self._inv_corr = tf.constant(self._inv_corr, dtype=tf.float64)

    def _con_term(self):
        con_pars = self.params.get_parameters([self.con_indices])
        chi2_cons = np.sum(((self.con_value - con_pars) / self.con_sigma) ** 2)
        return chi2_cons

    def _gauss_term(self, bin_pars):
        return tf.linalg.matmul(tf.transpose(bin_pars), tf.linalg.matmul(self._inv_corr, bin_pars))

    def chi2(self):
        diff = (self.expected_events_per_bin() - self.x_obs) / self.x_obs_errors
        bin_pars = tf.reshape(self.bin_pars, (self.shape[0] * self.shape[1],))
        chi2 = tf.tensordot(diff, diff, 1) + tf.tensordot(bin_pars, bin_pars, 1)
        # self._gauss_term(tf.reshape(self.bin_pars,(self.shape[0]*self.shape[1],1))) #+ self._con_term()
        return chi2

    def nll(self, pars):
        self.params.set_parameters(pars)
        exp_evts_per_bin = self.x_expected()
        poisson_term = np.sum(exp_evts_per_bin - self.x_obs -
                              xlogyx(self.x_obs, exp_evts_per_bin))

        nll = poisson_term + (self._gauss_term(self.bin_pars) + self._con_term()) / 2.
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
