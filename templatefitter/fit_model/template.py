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

        self._yield_parameter = None
        self._bin_parameters = None
        self._efficiency_parameter = None
        self._fraction_parameter = None

        self._template_index = None
        self._component_index = None
        self._channel_index = None

    # TODO: Needs rework
    def initialize_parameters(
            self,
            yield_index: int,
            bin_parameter_indices: List[int],
            efficiency_index: Optional[int] = None
    ):
        # TODO: Maybe this function should set the initial values for the parameters
        #       and set the indices from the parameter handler also in the respective TemplateParameters

        # TODO: Should set template, channel and component indices!

        # TODO: One should also be able to set the efficiency to a fixed value.

        self.yield_index = yield_index
        self.bin_parameter_indices = bin_parameter_indices
        if efficiency_index is not None:
            self.efficiency_index = efficiency_index

    @property
    def yield_parameter(self) -> Optional[TemplateParameter]:
        return self._yield_parameter

    @property
    def yield_index(self) -> Optional[int]:
        if self._yield_parameter is None:
            return None
        return self._yield_parameter.index

    @yield_index.setter
    def yield_index(self, index: int) -> None:
        if not isinstance(index, int):
            raise ValueError("Expected integer...")
        self._parameter_setter_checker(parameter=self._yield_parameter.index, parameter_name="yield_index")
        self._yield_parameter.index = index

    @property
    def efficiency_parameter(self) -> Optional[TemplateParameter]:
        return self._efficiency_parameter

    @property
    def efficiency_index(self) -> int:
        return self._efficiency_parameter.index

    @efficiency_index.setter
    def efficiency_index(self, index: int) -> None:
        if not isinstance(index, int):
            raise ValueError("Expected integer...")
        self._parameter_setter_checker(parameter=self._efficiency_parameter.index, parameter_name="efficiency_index")
        self._efficiency_parameter.index = index

    @property
    def fraction_parameter(self) -> Optional[TemplateParameter]:
        return self._fraction_parameter

    @property
    def fraction_index(self) -> int:
        return self._fraction_parameter.index

    @fraction_index.setter
    def fraction_index(self, index: int) -> None:
        if not isinstance(index, int):
            raise ValueError("Expected integer...")
        self._parameter_setter_checker(parameter=self._fraction_parameter.index, parameter_name="fraction_index")
        self._fraction_parameter.index = index

    @property
    def bin_parameters(self) -> List[Optional[TemplateParameter]]:
        return self._bin_parameters

    @property
    def bin_parameter_indices(self) -> List[Optional[int]]:
        return [None if bin_param is None else bin_param.index for bin_param in self._bin_parameters]

    @bin_parameter_indices.setter
    def bin_parameter_indices(self, indices: List[int]) -> None:
        if not (isinstance(indices, list) and all(isinstance(i, int) for i in indices)):
            raise ValueError("Expected list of integers...")
        assert len(indices) == self.num_bins_total, (len(indices), self.num_bins_total)
        assert len(indices) == len(self._bin_parameters), (len(indices), len(self._bin_parameters))
        for i, (bin_param, index) in enumerate(zip(self._bin_parameters, indices)):
            self._parameter_setter_checker(parameter=bin_param, parameter_name=f"bin_parameter_index_{i}")
            bin_param.index = index

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

    def fractions(self):
        # TODO: Should be able to calculate its own bin yields from parameters for plotting and so on...
        #       So maybe adapt this method to achieve this.
        per_bin_yields = (
                self.bin_counts_flattened
                * (1. + self._params.get_parameters_by_index(self.bin_parameter_indices) * self._relative_errors)
        )

        return per_bin_yields / np.sum(per_bin_yields)

    # TODO: Add some function to convert to hist for plotting...
