import logging
import numpy as np

from templatefitter.will_templates import AbstractTemplate

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["ChannelTemplate"]


class ChannelTemplate(AbstractTemplate):
    """
    Combines several will_templates according with both yields and efficiencies.
    There is the possibility of using shared yields between channeltemplates.
    """

    def __init__(
            self,
            name,
            variable,
            templates,
            params,
            initialyields,
            initialeffs,
            sharedyields='False',
            yieldnames=[],
            color=None,
            pretty_variable=None,
            pretty_label=None,
    ):
        super().__init__(name=name, params=params)

        self._templates = templates
        self._yield_indices = []
        self._eff_indices = []
        self._initial_yields = initialyields
        self._initial_effs = initialeffs
        self._shared_yields = 'False'
        self._yield_names = yieldnames
        self._init_params()

    def _init_params(self):
        """
        Register fractions / efficiencies as parameters
        """
        eff_names = ["efficiency_{}_{}".format(self.name, x) for x in templates.keys()]
        self._yield_indices = self._params.addParameters(self._initialpars, par_names)
        if self._shared_yields:
            self._yield_indices = self._initialyields
        else:
            if len(yield_names) > 0:
                yield_names = self._yield_names
            else:
                yield_names = ["yield_{}_{}".format(self.name, x) for x in templates.keys()]
            self._yield_indices = self._params.addParameters(self._initialpars, par_names)

    def get_pars(self):
        return self._params.getParameters(self._par_indices)

    def fractions(self):
        """
        Computes the multi-template bin-fractions using individual will_templates
        constructed together with the number of 
        """
        yields = self._params.getParameters(self._yield_indices)
        efficiencies = self._params.getParameters(self._eff_indices)
        fractions_per_template = np.array([template.fractions() for template in self.templates.values()])
        return (yields * efficiencies) @ fractions
