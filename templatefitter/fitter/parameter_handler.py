# TODO: Docstring
#  class which contains parameters

import numpy as np

__all__ = ["ParameterHandler"]


class ParameterHandler:
    """
    The class provides an interface for registering and storing parameters.
    """

    def __init__(self):
        """
        Initialise parameter array and dictionary.
        """
        # pars used for registration
        self._pars = []
        # access provided by n_pars
        self._n_pars = np.array([])
        self._pars_dict = {}

    def add_parameters(self, pars, names):
        indices = []
        for key, value in zip(names, pars):
            index = self.add_parameter(key, value, update='False')
            indices.append(index)
        self._n_pars = np.array(self._pars)
        return indices

    def add_parameter(self, name, parameter, update='True'):
        self._pars += [parameter]
        yield_index = len(self._pars) - 1
        self._pars_dict[name] = yield_index
        if update:
            self._n_pars = np.array(self._pars)
        return yield_index

    def get_parameters_by_slice(self, slicing):
        return self._n_pars[slicing[0]:slicing[1]]

    def get_parameters_by_index(self, indices):
        return self._n_pars[indices]

    def get_parameters(self):
        return self._pars_dict

    def get_parameter_names(self):
        return self._pars_dict.keys()

    def set_parameters(self, pars):
        self._n_pars[:] = pars

    def get_index(self, name):
        return self._pars_dict[name]
