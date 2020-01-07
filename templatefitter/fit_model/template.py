"""
Template Class to be used in fit models.
"""

import logging
import numpy as np

from typing import Optional, List, Tuple

from templatefitter.fit_model.parameter_handler import ParameterHandler, TemplateParameter

from templatefitter.binned_distributions.weights import WeightsInputType
from templatefitter.binned_distributions.binning import BinsInputType, ScopeInputType, LogScaleInputType
from templatefitter.binned_distributions.binned_distribution import BinnedDistribution, DataInputType, \
    SystematicsInputType, DataColumnNamesInput

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["Template"]


# TODO: Check TODOs below!
class Template(BinnedDistribution):
    def __init__(
            self,
            name: str,
            process_name: str,
            dimensions: int,
            bins: BinsInputType,
            scope: ScopeInputType,
            params: ParameterHandler,
            data_column_names: DataColumnNamesInput,
            latex_label: Optional[str] = None,
            color: Optional[str] = None,
            data: Optional[DataInputType] = None,
            weights: WeightsInputType = None,
            systematics: SystematicsInputType = None,
            log_scale_mask: LogScaleInputType = False
    ):
        super().__init__(
            bins=bins,
            dimensions=dimensions,
            scope=scope,
            log_scale_mask=log_scale_mask,
            name=name,
            data=data,
            weights=weights,
            systematics=systematics,
            data_column_names=data_column_names
        )
        self._params = params
        self._process_name = process_name
        self._serial_number = None
        self._component_serial_number = None
        self._use_other_systematics = True

        self._color = color
        if latex_label is None:
            self._latex_label = process_name
        else:
            self._latex_label = latex_label

        self._yield_parameter = None
        self._bin_nuisance_parameters = None
        self._efficiency_parameter = None
        self._fraction_parameter = None

        self._template_index = None
        self._component_index = None
        self._channel_index = None

    def initialize_parameters(
            self,
            yield_parameter: TemplateParameter,
            bin_nuisance_parameters: Optional[List[TemplateParameter]]
    ) -> None:
        if self._yield_parameter is not None or self._bin_nuisance_parameters is not None:
            raise RuntimeError(
                f"Trying to reset template's "
                f"{'yield_parameter' if self._yield_parameter is None else 'bin_nuisance_parameters'}..."
            )
        if not isinstance(yield_parameter, TemplateParameter):
            raise ValueError(f"Argument 'yield_parameter' must be of type TemplateParameter!\n"
                             f"You provided an object of type {type(yield_parameter)}...")
        if not yield_parameter.parameter_type == ParameterHandler.yield_parameter_type:
            raise ValueError(f"Expected TemplateParameter of parameter_type 'yield', "
                             f"but received one with parameter_type {yield_parameter.parameter_type}!")
        self._yield_parameter = yield_parameter

        if bin_nuisance_parameters is None:
            self._bin_nuisance_parameters = None
            return

        if not (isinstance(bin_nuisance_parameters, list) and len(bin_nuisance_parameters) == self.num_bins_total):
            raise ValueError(f"The argument 'bin_nuisance_parameters' must be a list with {self.num_bins_total} "
                             f"TemplateParameters of type {ParameterHandler.bin_nuisance_parameter_type}, but "
                             + f"is an object of type {type(bin_nuisance_parameters)}."
                             if not isinstance(bin_nuisance_parameters, list)
                             else f" has only {len(bin_nuisance_parameters)} elements while the template has "
                                  f"{self.num_bins_total} bins.")
        if not all(isinstance(p, TemplateParameter) for p in bin_nuisance_parameters):
            raise ValueError("The parameter bin_nuisance_parameters must be a list of TemplateParameters, "
                             "but some elements are not TemplateParameters.")
        if not all(p.parameter_type == ParameterHandler.bin_nuisance_parameter_type for p in bin_nuisance_parameters):
            raise ValueError(f"The parameter bin_nuisance_parameters must be a list of TemplateParameters of type "
                             f"{ParameterHandler.bin_nuisance_parameter_type}.")
        self._bin_nuisance_parameters = bin_nuisance_parameters

    @property
    def process_name(self) -> str:
        return self._process_name

    @property
    def latex_label(self) -> str:
        return self._latex_label

    @property
    def color(self) -> str:
        return self._color

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
        self._template_parameter_setter_check(
            input_parameter=efficiency_parameter,
            parameter_type=ParameterHandler.efficiency_parameter_type,
            parameter_name="efficiency_parameter"
        )
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
        self._template_parameter_setter_check(
            input_parameter=fraction_parameter,
            parameter_type=ParameterHandler.fraction_parameter_type,
            parameter_name="fraction_parameter"
        )
        self._fraction_parameter = fraction_parameter

    @staticmethod
    def _template_parameter_setter_check(
            input_parameter: TemplateParameter,
            parameter_type: str,
            parameter_name: str
    ) -> None:
        assert parameter_type in ParameterHandler.parameter_types, \
            f"parameter_type must be one of {ParameterHandler.parameter_types}, you provided {parameter_type}!"

        if not isinstance(input_parameter, TemplateParameter):
            raise ValueError(f"The {parameter_name} can only be set to a TemplateParameter. "
                             f"You provided an object of type {type(input_parameter)}!")
        if input_parameter.parameter_type != parameter_type:
            raise ValueError(f"The {parameter_name} can only be set to a TemplateParameter of type {parameter_type}."
                             f"However, the provided TemplateParameter is of parameter_type "
                             f"{input_parameter.parameter_type}...")

    @property
    def fraction_index(self) -> Optional[int]:
        if self._fraction_parameter is None:
            return None
        return self._fraction_parameter.index

    @property
    def bin_nuisance_parameters(self) -> Optional[List[TemplateParameter]]:
        return self._bin_nuisance_parameters

    @property
    def bin_nuisance_parameter_indices(self) -> Optional[List[int]]:
        if self._bin_nuisance_parameters is None:
            return None
        return [bin_nuisance_param.index for bin_nuisance_param in self._bin_nuisance_parameters]

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

    @property
    def use_other_systematics(self) -> bool:
        return self._use_other_systematics

    @use_other_systematics.setter
    def use_other_systematics(self, boolean_value) -> None:
        self._use_other_systematics = boolean_value

    # TODO: Needs work...
    def expected_bin_counts(self):
        pass
        # TODO!

    def fractions(self):
        # TODO: Should be able to calculate its own bin yields from parameters for plotting and so on...
        #       So maybe adapt this method to achieve this.
        per_bin_yields = (
                self.bin_counts_flattened
                * (1. + self._params.get_parameters_by_index(self.bin_parameter_indices) * self._relative_errors)
        )

        return per_bin_yields / np.sum(per_bin_yields)

    # TODO: Add some function to convert to hist for plotting...
