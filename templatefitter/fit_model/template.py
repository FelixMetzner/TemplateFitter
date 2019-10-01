"""
Template Class to be used in fit models.
"""

import logging
import numpy as np

from typing import List

from templatefitter.fit_model.parameter_handler import ParameterHandler
from templatefitter.binned_distributions.binning import BinsInputType, ScopeInputType
from templatefitter.binned_distributions.binned_distribution import BinnedDistribution

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
            yield_index: int,
            bin_parameter_indices: List[int],
            efficiency_index: int,
            channel_index: int,
            params: ParameterHandler
    ):
        super().__init__(bins=bins, dimensions=dimensions, scope=scope, name=name)
        self._params = params

        self._yield_index = yield_index
        self._bin_parameter_indices = bin_parameter_indices
        self._efficiency_index = efficiency_index
        self._channel_index = channel_index

        self._statistics = None

    @property
    def yield_index(self) -> int:
        return self._yield_index

    @property
    def bin_parameter_indices(self) -> List[int]:
        return self._bin_parameter_indices

    @property
    def efficiency_index(self) -> int:
        return self._efficiency_index

    @property
    def channel_index(self) -> int:
        return self._channel_index

    @property
    def bin_counts_flattened(self) -> np.ndarray:
        return self.bin_counts.flatten()

    def fractions(self):
        per_bin_yields = (
                self.bin_counts_flattened
                * (1. + self._params.get_parameters_by_index(self.bin_parameter_indices) * self._relative_errors)
        )

        return per_bin_yields / np.sum(per_bin_yields)

    # TODO: Add some function to convert to hist for plotting...
