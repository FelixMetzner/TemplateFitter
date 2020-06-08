"""
Template Class to be used in fit models.
"""

import copy
import logging
import numpy as np

from typing import Optional, List, Tuple

from templatefitter.fit_model.parameter_handler import ParameterHandler, TemplateParameter

from templatefitter.binned_distributions.weights import WeightsInputType
from templatefitter.binned_distributions.binning import BinsInputType, ScopeInputType, LogScaleInputType
from templatefitter.binned_distributions.binned_distribution import BinnedDistributionFromData, DataInputType, \
    DataColumnNamesInput
from templatefitter.binned_distributions.systematics import SystematicsInputType

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["Template"]


class Template(BinnedDistributionFromData):
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
        self._other_fraction_parameters = None  #: type Optional[List[TemplateParameter]]

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

    @property
    def fraction_index(self) -> Optional[int]:
        if self._fraction_parameter is None:
            return None
        return self._fraction_parameter.index

    @property
    def other_fraction_parameters(self) -> Optional[List[TemplateParameter]]:
        return self._other_fraction_parameters

    @other_fraction_parameters.setter
    def other_fraction_parameters(self, other_fraction_parameters: List[TemplateParameter]) -> None:
        if self._other_fraction_parameters is not None:
            raise RuntimeError(f"Trying to reset list of other fraction parameters of template {self.name}.")

        for other_fraction_param in other_fraction_parameters:
            self._template_parameter_setter_check(
                input_parameter=other_fraction_param,
                parameter_type=ParameterHandler.fraction_parameter_type,
                parameter_name="elements of other_fraction_parameters"
            )
        self._other_fraction_parameters = other_fraction_parameters

    @property
    def other_fractions_indices(self) -> Optional[List[int]]:
        if self._other_fraction_parameters is None:
            return None
        return [other_fraction_param.index for other_fraction_param in self._other_fraction_parameters]

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

    def expected_bin_counts(self, use_initial_values: bool = False) -> np.ndarray:
        """
        This function is only used to plot the individual templates in the FitResultPlot.
        During the fit, all bin counts are handled in one matrix.
        :return: np.ndarray containing the bin counts for this template
        """
        template_shape = self.get_template_shape_for_expected_bin_counts(use_initial_values=use_initial_values)

        if use_initial_values:
            template_yield = self.yield_parameter.initial_value
            template_efficiency = self.efficiency_parameter.initial_value
        else:
            template_yield = self.params.get_parameters_by_index(indices=self.yield_index)
            template_efficiency = self.params.get_parameters_by_index(indices=self.efficiency_index)

        template_fraction = self.get_fraction_for_expected_bin_counts(use_initial_values=use_initial_values)

        return template_shape * template_yield * template_fraction * template_efficiency

    def expected_bin_errors_squared(self, use_initial_values: bool = False) -> np.ndarray:
        """
        This function is only used to plot the uncertainty of the individual templates in the FitResultPlot.
        During the fit, the uncertainties are handled via matrices and nuisance parameters.
        :return: np.ndarray containing the squared bin uncertainties for this template
        """
        relative_uncertainties = self._get_relative_uncertainties_for_plotting(use_stat_only=False)
        return self.expected_bin_counts(use_initial_values=use_initial_values) * relative_uncertainties

    def get_template_shape_for_expected_bin_counts(self, use_initial_values: bool = False) -> np.ndarray:
        template_bin_count = copy.copy(self.bin_counts)

        if not use_initial_values and self.bin_nuisance_parameters is not None:
            # TODO: This has to be updated once different versions of the handling of nuisance parameters are in place.
            nuisance_parameters = self.params.get_parameters_by_index(indices=self.bin_nuisance_parameter_indices)
            nuisance_parameters = np.reshape(nuisance_parameters, newshape=self.num_bins)
            relative_shape_uncertainties = self._get_relative_uncertainties_for_plotting()
            assert template_bin_count.shape == relative_shape_uncertainties.shape, \
                (template_bin_count.shape, relative_shape_uncertainties.shape)
            assert template_bin_count.shape == nuisance_parameters.shape, \
                (template_bin_count.shape, nuisance_parameters.shape)
            template_bin_count *= 1. + nuisance_parameters * relative_shape_uncertainties

        if template_bin_count.sum() == 0.:
            return template_bin_count
        template_shape = template_bin_count / template_bin_count.sum()
        return template_shape

    def _get_relative_uncertainties_for_plotting(self, use_stat_only: bool = False) -> np.ndarray:
        template_bin_count = copy.copy(self.bin_counts)

        stat_errors_sq = self.bin_errors_sq
        if self.use_other_systematics and not use_stat_only:
            sys_errors_sq = np.reshape(np.diag(self.bin_covariance_matrix), newshape=self.num_bins)
            assert stat_errors_sq.shape == sys_errors_sq.shape, (stat_errors_sq.shape, sys_errors_sq.shape)
            uncertainties_sq = stat_errors_sq + sys_errors_sq
        else:
            uncertainties_sq = stat_errors_sq

        uncertainties = np.sqrt(uncertainties_sq)
        relative_shape_uncertainties = np.divide(
            uncertainties,
            template_bin_count,
            out=np.zeros_like(uncertainties),
            where=template_bin_count != 0.
        )
        return relative_shape_uncertainties

    def get_fraction_for_expected_bin_counts(self, use_initial_values: bool = False) -> float:
        template_fraction = 1.
        if self.fraction_parameter is not None:
            assert self.other_fraction_parameters is None
            if use_initial_values:
                template_fraction = self.fraction_parameter.initial_value
            else:
                template_fraction = self.params.get_parameters_by_index(indices=self.fraction_index)
        if self.other_fraction_parameters is not None:
            assert self.fraction_parameter is None
            if use_initial_values:
                for other_fraction_parameter in self.other_fraction_parameters:
                    template_fraction -= other_fraction_parameter.initial_value
            else:
                for other_fraction_index in self.other_fractions_indices:
                    template_fraction -= self.params.get_parameters_by_index(indices=other_fraction_index)

        assert template_fraction >= 0., (template_fraction, self.fraction_index, self.other_fractions_indices)
        assert template_fraction <= 1., (template_fraction, self.fraction_index, self.other_fractions_indices)

        return template_fraction

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
