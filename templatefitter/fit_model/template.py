"""
Template Class to be used in fit models.
"""

import logging
import numpy as np

from typing import Optional, List, Tuple

from templatefitter.binned_distributions.binning import BinsInputType, ScopeInputType
from templatefitter.fit_model.parameter_handler import ParameterHandler, TemplateParameter
from templatefitter.binned_distributions.binned_distribution import BinnedDistribution, DataColumnNamesInput

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["Template"]


class Template(BinnedDistribution):
    def __init__(
            self,
            name: str,
            dimensions: int,
            bins: BinsInputType,
            scope: ScopeInputType,
            params: ParameterHandler,
            data_column_names: DataColumnNamesInput
    ):
        super().__init__(bins=bins, dimensions=dimensions, scope=scope, name=name, data_column_names=data_column_names)
        self._params = params
        self._serial_number = None
        self._component_serial_number = None

        self._yield_parameter = None
        self._bin_uncert_parameters = None
        self._efficiency_parameter = None
        self._fraction_parameter = None

        self._template_index = None
        self._component_index = None
        self._channel_index = None

    def initialize_parameters(
            self,
            yield_parameter: TemplateParameter,
            bin_uncert_parameters: List[TemplateParameter]
    ):
        if self._yield_parameter is not None or self._bin_uncert_parameters is not None:
            raise RuntimeError(f"Trying to reset template's "
                               f"{'yield_parameter' if self._yield_parameter is None else 'bin_uncert_parameter'}...")
        if not isinstance(yield_parameter, TemplateParameter):
            raise ValueError(f"Argument 'yield_parameter' must be of type TemplateParameter!\n"
                             f"You provided an object of type {type(yield_parameter)}...")
        if not yield_parameter.parameter_type == ParameterHandler.yield_parameter_type:
            raise ValueError(f"Expected TemplateParameter of parameter_type 'yield', "
                             f"but received one with parameter_type {yield_parameter.parameter_type}!")
        self._yield_parameter = yield_parameter

        if not (isinstance(bin_uncert_parameters, list) and len(bin_uncert_parameters) == self.num_bins_total):
            raise ValueError(f"The argument 'bin_uncert_parameters' must be a list with {self.num_bins_total} "
                             f"TemplateParameters of type {ParameterHandler.bin_uncert_parameter_type}, but "
                             + f"is an object of type {type(bin_uncert_parameters)}."
                             if not isinstance(bin_uncert_parameters, list)
                             else f" has only {len(bin_uncert_parameters)} elements while the template has "
                                  f"{self.num_bins_total} bins.")
        if not all(isinstance(p, TemplateParameter) for p in bin_uncert_parameters):
            raise ValueError("The parameter bin_uncert_parameters must be a list of TemplateParameters, "
                             "but some elements are not TemplateParameters.")
        if not all(p.parameter_type == ParameterHandler.bin_uncert_parameter_type for p in bin_uncert_parameters):
            raise ValueError(f"The parameter bin_uncert_parameters must be a list of TemplateParameters of type "
                             f"{ParameterHandler.bin_uncert_parameter_type}.")
        self._bin_uncert_parameters = bin_uncert_parameters

    @property
    def serial_number(self) -> int:
        assert self._serial_number is not None
        return self._serial_number

    @serial_number.setter
    def serial_number(self, serial_number: int) -> None:
        if self._serial_number is not None:
            raise RuntimeError(f"Trying to reset template serial number from {self._serial_number} to {serial_number}!")
        self._serial_number = serial_number

    @property
    def component_serial_number(self) -> int:
        assert self._component_serial_number is not None
        return self._component_serial_number

    @component_serial_number.setter
    def component_serial_number(self, component_serial_number: int) -> None:
        if self._component_serial_number is not None:
            raise RuntimeError(f"Trying to reset template's component serial number from "
                               f"{self._component_serial_number} to {component_serial_number}!")
        self._component_serial_number = component_serial_number

    @property
    def yield_parameter(self) -> Optional[TemplateParameter]:
        return self._yield_parameter

    @property
    def yield_index(self) -> Optional[int]:
        if self._yield_parameter is None:
            return None
        return self._yield_parameter.index

    @property
    def efficiency_parameter(self) -> Optional[TemplateParameter]:
        return self._efficiency_parameter

    @efficiency_parameter.setter
    def efficiency_parameter(self, efficiency_parameter: TemplateParameter) -> None:
        if self._efficiency_parameter is not None:
            raise RuntimeError(f"Trying to reset efficiency_parameter of template {self.name}.")
        if not isinstance(efficiency_parameter, TemplateParameter):
            raise ValueError(f"The efficiency_parameter can only be set to a TemplateParameter. "
                             f"You provided an object of type {type(efficiency_parameter)}!")
        if efficiency_parameter.parameter_type != ParameterHandler.efficiency_parameter_type:
            raise ValueError(f"The efficiency_parameter can only be set to a TemplateParameter "
                             f"of type {ParameterHandler.efficiency_parameter_type}. However, the provided "
                             f"TemplateParameter is of parameter_type {efficiency_parameter.parameter_type}...")
        self._efficiency_parameter = efficiency_parameter

    @property
    def efficiency_index(self) -> Optional[int]:
        if self._efficiency_parameter is None:
            return None
        return self._efficiency_parameter.index

    @property
    def fraction_parameter(self) -> Optional[TemplateParameter]:
        return self._fraction_parameter

    @fraction_parameter.setter
    def fraction_parameter(self, fraction_parameter: TemplateParameter) -> None:
        if self._fraction_parameter is not None:
            raise RuntimeError(f"Trying to reset fraction parameter of template {self.name}.")
        if not isinstance(fraction_parameter, TemplateParameter):
            raise ValueError(f"The fraction_parameter can only be set to a TemplateParameter. "
                             f"You provided an object of type {type(fraction_parameter)}!")
        if fraction_parameter.parameter_type != ParameterHandler.fraction_parameter_type:
            raise ValueError(f"The fraction_parameter can only be set to a TemplateParameter "
                             f"of type {ParameterHandler.fraction_parameter_type}. However, the provided "
                             f"TemplateParameter is of parameter_type {fraction_parameter.parameter_type}...")
        self._fraction_parameter = fraction_parameter

    @property
    def fraction_index(self) -> Optional[int]:
        if self._fraction_parameter is None:
            return None
        return self._fraction_parameter.index

    @property
    def bin_uncert_parameters(self) -> Optional[List[TemplateParameter]]:
        return self._bin_uncert_parameters

    @property
    def bin_uncert_parameter_indices(self) -> Optional[List[int]]:
        if self._bin_uncert_parameters is None:
            return None
        return [bin_uncert_param.index for bin_uncert_param in self._bin_uncert_parameters]

    @property
    def global_template_identifier(self) -> Tuple[int, int, int]:
        return self.channel_index, self.component_index, self.template_index

    @property
    def template_index(self) -> int:
        assert self._template_index is not None
        return self._template_index

    @template_index.setter
    def template_index(self, index: int) -> None:
        if not isinstance(index, int):
            raise ValueError("Expected integer...")
        self._parameter_setter_checker(parameter=self._template_index, parameter_name="template_index")
        self._template_index = index

    @property
    def component_index(self) -> int:
        assert self._component_index is not None
        return self._component_index

    @component_index.setter
    def component_index(self, index: int) -> None:
        if not isinstance(index, int):
            raise ValueError("Expected integer...")
        self._parameter_setter_checker(parameter=self._component_index, parameter_name="component_index")
        self._component_index = index

    @property
    def channel_index(self) -> int:
        assert self._channel_index is not None
        return self._channel_index

    @channel_index.setter
    def channel_index(self, index: int) -> None:
        if not isinstance(index, int):
            raise ValueError("Expected integer...")
        self._parameter_setter_checker(parameter=self._channel_index, parameter_name="channel_index")
        self._channel_index = index

    @property
    def bin_counts_flattened(self) -> np.ndarray:
        return self.bin_counts.flatten()

    def _parameter_setter_checker(self, parameter, parameter_name):
        if parameter is not None:
            name_info = "" if self.name is None else f" with name '{self.name}'"
            raise RuntimeError(f"Trying to reset {parameter_name} for template{name_info}.")

    @property
    def params(self) -> ParameterHandler:
        return self._params

    # TODO: Needs work...
    def fractions(self):
        # TODO: Should be able to calculate its own bin yields from parameters for plotting and so on...
        #       So maybe adapt this method to achieve this.
        per_bin_yields = (
                self.bin_counts_flattened
                * (1. + self._params.get_parameters_by_index(self.bin_parameter_indices) * self._relative_errors)
        )

        return per_bin_yields / np.sum(per_bin_yields)

    # TODO: Add some function to convert to hist for plotting...
