"""
Parameter Handler
    Class for managing parameters of all templates contained within a fit model.
"""
import logging
import numpy as np
from typing import Optional, Union, List, Tuple, Dict

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["ParameterHandler"]


class ParameterHandler:
    def __init__(self):
        self._pars = []
        self._np_pars = np.array([])
        self._pars_dict = {}
        self._inverted_pars_dict = None

    # TODO: Differentiate between FixedParameter and FloatingParameter
    #       - Maybe add respective classes, which should inherit from a Parameter base class.
    #       - Use these parameter classes in templates and components (maybe there should only be
    #         one parameter class then... we will see)
    #       - Parameter class should have knowledge of the initial value of its parameter, of the current one
    #         and should maybe allow to temporarily overwrite the parameter for tests/plotting or whatever.

    def add_parameters(self, pars: Union[np.ndarray, List[float]], names: List[str]) -> List[int]:
        self._check_input_pars_and_names(pars=pars, names=names)

        indices = []
        for name, parameter in zip(names, pars):
            index = self.add_parameter(parameter=parameter, name=name, update=False)
            indices.append(index)

        assert len(indices) == len(pars)
        self._np_pars = np.array(self._pars)

        return indices

    def add_parameter(self, parameter: float, name: str, update: bool = True) -> int:
        assert name not in self._pars_dict.keys(), (name, self._pars_dict.keys())

        parameter_index = len(self._pars)
        self._pars.append(parameter)

        assert parameter_index not in self._pars_dict.values(), (parameter_index, self._pars_dict.values())
        self._pars_dict[name] = parameter_index
        self._inverted_pars_dict = None

        if update:
            self._np_pars = np.array(self._pars)

        return parameter_index

    def get_index(self, name: str) -> int:
        return self._pars_dict[name]

    def get_name(self, index: int) -> str:
        if self._inverted_pars_dict is None:
            self._inverted_pars_dict = {v: k for k, v in self._pars_dict.items()}
        return self._pars_dict[index]

    def get_parameters_by_slice(self, slicing: Tuple[Optional[int], Optional[int]]) -> np.ndarray:
        return self._np_pars[slicing[0]:slicing[1]]

    def get_parameters_by_index(self, indices: Union[int, List[int]]) -> Union[np.ndarray, float]:
        return self._np_pars[indices]

    def get_parameter_dictionary(self) -> Dict[str, int]:
        return self._pars_dict

    def get_parameter_names(self) -> List[str]:
        return list(self._pars_dict.keys())

    def get_parameters(self) -> np.ndarray:
        return self._np_pars

    def set_parameters(self, pars: np.ndarray) -> None:
        if len(pars.shape) != 1:
            raise ValueError(f"Parameter 'pars' must be 1 dimensional, but has shape {pars.shape}...")
        if len(pars) != len(self._np_pars):
            raise ValueError(f"Length of provided parameter array (= {len(pars)}) "
                             f"does not match the length of the existing parameter array (= {len(self._np_pars)})")
        self._np_pars[:] = pars

    def set_parameters_and_names(self, pars: np.ndarray, names: List[str]) -> None:
        self._check_input_pars_and_names(pars=pars, names=names)
        self._np_pars[:] = pars
        self._pars_dict = {name: index for index, name in enumerate(names)}
        self._inverted_pars_dict = None

    @staticmethod
    def _check_input_pars_and_names(pars: Union[np.ndarray, List[float]], names: List[str]) -> None:
        if isinstance(pars, np.ndarray) and len(pars.shape) != 1:
            raise ValueError(f"Parameter 'pars' must be 1 dimensional, but has shape {pars.shape}...")
        if not len(pars) == len(names):
            raise ValueError(f"Length of 'pars' and 'names' should be the same, but is {len(pars)} != {len(names)}!")
        if not len(names) == len(set(names)):
            raise ValueError(f"Entries of 'names' should be unique! You provided:\n{names}")
