"""
Class providing a generalized weights object.
"""

import logging
import numpy as np
import pandas as pd
from typing import Union, Optional, Any

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "Weights",
    "WeightsInputType"
]

WeightsInputType = Union[None, float, str, pd.Series, np.ndarray]


class Weights:
    def __init__(
            self,
            weight_input: WeightsInputType = None,
            data: Union[None, pd.Series, np.ndarray] = None,
            data_input: Optional[pd.DataFrame] = None
    ) -> None:
        self._weights = None
        self.check_input_type(weight_input=weight_input)
        self._init_weights(weight_input=weight_input, data=data, data_input=data_input)

    def _init_weights(
            self,
            weight_input: WeightsInputType,
            data: Union[None, pd.Series, np.ndarray],
            data_input: Optional[pd.DataFrame]
    ) -> None:
        if weight_input is None:
            assert data is not None
            self._weights = np.ones(len(data))
        elif isinstance(weight_input, float):
            assert data is not None
            self._weights = np.ones(len(data)) * weight_input
        elif isinstance(weight_input, str):
            assert data_input is not None
            assert isinstance(data_input, pd.DataFrame), type(data_input)
            assert weight_input in data_input.columns
            self._weights = data_input[weight_input].values
        elif isinstance(weight_input, pd.Series):
            self._weights = weight_input.values
        elif isinstance(weight_input, np.ndarray):
            self._weights = weight_input
        else:
            raise RuntimeError(f"Got unexpected type for weights: {type(weight_input)}.\n"
                               f"Should be one of None, str, float, pd.Series, np.ndarray.")

        assert isinstance(self._weights, np.ndarray)

    @classmethod
    def obtain_weights(
            cls,
            weight_input: WeightsInputType,
            data: Union[None, pd.Series, np.ndarray],
            data_input: Optional[pd.DataFrame]
    ) -> np.ndarray:
        instance = cls(weight_input=weight_input, data=data, data_input=data_input)
        return instance.get_weights()

    def get_weights(self) -> np.ndarray:
        return self._weights

    @staticmethod
    def check_input_type(weight_input: Any) -> None:
        acceptable_types = tuple([type(None), float, str, pd.Series, np.ndarray])
        assert isinstance(weight_input, acceptable_types), (type(weight_input), acceptable_types)
