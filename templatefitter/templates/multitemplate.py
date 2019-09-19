import logging
import numpy as np

from scipy.linalg import block_diag

from templatefitter.templates import AbstractTemplate

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "MultiTemplate",
    "MultiNormTemplate"
]


class MultiTemplate(AbstractTemplate):
    """ combines several templates according to fractions.
    This produces a new pdf.
    """

    def __init__(
            self,
            name,
            variable,
            templates,
            params,
            initialpars,
            color=None,
            pretty_variable=None,
            pretty_label=None,
    ):
        super(MultiTemplate, self).__init__(name=name, params=params)

        self._templates = templates
        self._par_indices = []
        self._initial_pars = initialpars
        self._init_params()
        self._num_templates = len(templates)

        self.pretty_variable = pretty_variable
        self.color = color
        self.pretty_label = pretty_label

    def _init_params(self):
        """ Register fractions / efficiencies as parameters """
        par_names = ["efficiency_{}".format(x) for x in templates.keys()]
        self._par_indices = self._params.addParameters(self._initial_pars, par_names)

    def get_pars(self):
        return self._params.getParameters(self._par_indices)

    def get_bin_pars(self):
        return np.array([template.get_bin_pars() for template in self._templates.values()])

    def fractions(self):
        """
        Computes the multi-template bin-fractions using individual templates
        constructed together with the number of TODO ???
        """
        pars = self._params.getParameters(self._par_indices)
        fractions_per_template = np.array([template.fractions() for template in self.templates.values()])
        return pars @ fractions_per_template


class MultiNormTemplate(AbstractTemplate):
    """
    Combines several templates according to fractions.
    This produces a new pdf.
    """

    def __init__(
            self,
            name,
            templates,
            params,
            initial_pars,
            color=None,
            pretty_variable=None,
            pretty_label=None,
    ):
        super(MultiNormTemplate, self).__init__(name=name, params=params)

        self._templates = templates
        self._bins = next(iter(self._templates.values()))._bins
        self._par_indices = []
        self._initial_pars = initial_pars
        self._init_params()
        self._num_templates = len(templates)

        self.pretty_variable = pretty_variable
        self.color = color
        self.pretty_label = pretty_label

    def _init_params(self):
        """ Register fractions / efficiencies as parameters """
        par_names = ["fraction_{}".format(x) for x in self._initial_pars.keys()]
        self._par_indices = self._params.addParameters(self._initial_pars.values(), par_names)

    def get_pars(self):
        return self._params.getParameters(self._par_indices)

    def get_bin_pars(self):
        return np.concatenate([template.get_bin_pars() for template in self._templates.values()])

    def fractions(self):
        """
        Computes the multi-template bin-fractions using individual templates
        constructed together with the number of 
        """
        pars = self._params.getParameters(self._par_indices)
        pars = np.append(pars, 1 - np.sum(pars))
        fractions_per_template = np.array(
            [template.fractions() for template
             in self._templates.values()]
        )
        return pars @ fractions_per_template

    def reshaped_fractions(self):
        """
        Calculates the expected number of events per bin using the current yield value and nuisance parameters.
        Shape is (`num_bins`,).
        """
        return self.fractions().reshape(self.shape())

    def all_fractions(self):
        """
        Computes the multi-template bin-fractions using individual templates
        constructed together with the number of 
        """
        pars = self._params.getParameters(self._par_indices)
        pars = np.append(pars, 1 - np.sum(pars))
        fractions_per_template = np.array(
            [template.fractions() for template
             in self._templates.values()]
        )
        return np.concatenate(pars[:, np.newaxis] * fractions_per_template)

    def inv_corr_mat(self):
        inv_corr_mats = [template.inv_corr_mat() for template
                         in self._templates.values()]
        return block_diag(*inv_corr_mats)

    def shape(self):
        return next(iter(self._templates.values())).shape()

    def bin_mids(self):
        return next(iter(self._templates.values())).bin_mids()

    def bin_edges(self):
        return next(iter(self._templates.values())).bin_edges()

    def bin_widths(self):
        return next(iter(self._templates.values())).bin_widths()

    def errors(self):
        """
        numpy.ndarray: Total uncertainty per bin.
        This value is the product of the relative uncertainty per bin and the current bin values.
        Shape is (`num_bins`,).
        """
        pars = self._params.getParameters(self._par_indices)
        pars = np.append(pars, 1 - np.sum(pars))
        uncertainties_sq = [
            (par * template.fractions() * template.errors()) ** 2
            for par, template in zip(pars, self._templates.values())
        ]
        total_uncertainty = np.sqrt(np.sum(np.array(uncertainties_sq), axis=0))
        return total_uncertainty

    def labels(self):
        return [template.labels()[0] for template in self._templates.values()]

    def colors(self):
        return [template.colors()[0] for template in self._templates.values()]
