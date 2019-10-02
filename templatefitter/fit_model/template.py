"""
Template Class to be used in fit models.
"""

import logging
import numpy as np

from typing import List, Optional

from templatefitter.fit_model.parameter_handler import ParameterHandler
from templatefitter.binned_distributions.binning import BinsInputType, ScopeInputType
from templatefitter.binned_distributions.binned_distribution import BinnedDistribution, DataColumnNamesInput

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["Template"]


# TODO: Check __init__ once all files have been refactored


class Template(BinnedDistribution):
    def __init__(
            self,
            name: str,
            dimensions: int,
            bins: BinsInputType,
            scope: ScopeInputType,
            yield_index: Optional[int],
            bin_parameter_indices: Optional[List[int]],
            efficiency_index: Optional[int],
            component_index: Optional[int],
            channel_index: Optional[int],
            params: ParameterHandler,
            data_column_names: DataColumnNamesInput
    ):
        super().__init__(bins=bins, dimensions=dimensions, scope=scope, name=name, data_column_names=data_column_names)
        self._params = params

        self._yield_index = yield_index
        self._bin_parameter_indices = bin_parameter_indices
        self._efficiency_index = efficiency_index
        self._component_index = component_index
        self._channel_index = channel_index

    @property
    def yield_index(self) -> int:
        return self._yield_index

    @yield_index.setter
    def yield_index(self, index: int) -> None:
        if not isinstance(index, int):
            raise ValueError("Expected integer...")
        self._parameter_setter_checker(parameter=self._yield_index, parameter_name="yield_index")
        self._yield_index = index

    @property
    def bin_parameter_indices(self) -> List[int]:
        return self._bin_parameter_indices

    @bin_parameter_indices.setter
    def bin_parameter_indices(self, indices: List[int]) -> None:
        if not (isinstance(indices, list) and all(isinstance(i, int) for i in indices)):
            raise ValueError("Expected list of integers...")
        self._parameter_setter_checker(parameter=self._bin_parameter_indices, parameter_name="bin_parameter_indices")
        self._bin_parameter_indices = indices

    @property
    def efficiency_index(self) -> int:
        return self._efficiency_index

    @efficiency_index.setter
    def efficiency_index(self, index: int) -> None:
        if not isinstance(index, int):
            raise ValueError("Expected integer...")
        self._parameter_setter_checker(parameter=self._efficiency_index, parameter_name="efficiency_index")
        self._efficiency_index = index

    @property
    def component_index(self) -> int:
        return self._component_index

    @component_index.setter
    def component_index(self, index: int) -> None:
        if not isinstance(index, int):
            raise ValueError("Expected integer...")
        self._parameter_setter_checker(parameter=self._component_index, parameter_name="component_index")
        self._component_index = index

    @property
    def channel_index(self) -> int:
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

    def fractions(self):
        # TODO: Should be able to calculate its own bin yields from parameters for plotting and so on...
        #       So maybe adapt this method to achieve this.
        per_bin_yields = (
                self.bin_counts_flattened
                * (1. + self._params.get_parameters_by_index(self.bin_parameter_indices) * self._relative_errors)
        )

        return per_bin_yields / np.sum(per_bin_yields)

    # TODO: Add some function to convert to hist for plotting...
