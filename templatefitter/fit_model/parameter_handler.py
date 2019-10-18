"""
Parameter Handler
    Class for managing parameters of all templates contained within a fit model.
"""
import logging
import numpy as np

from abc import ABC, abstractmethod
from typing import Optional, Union, List, Tuple, Dict, NamedTuple

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["ParameterHandler", "TemplateParameter", "ModelParameter"]


class ParameterInfo(NamedTuple):
    index: int
    name: str
    parameter_type: str
    floating: bool
    initial_value: float

    def info_as_string_list(self) -> List[str]:
        return [
            f"index: {self.index}",
            f"name: {self.name}",
            f"parameter_type: {self.parameter_type}",
            f"floating: {self.floating}",
            f"initial_value: {self.initial_value}",
        ]


class ParameterHandler:
    yield_parameter_type = "yield"
    fraction_parameter_type = "fraction"
    efficiency_parameter_type = "efficiency"
    parameter_types = [
        yield_parameter_type,
        fraction_parameter_type,
        efficiency_parameter_type
    ]

    def __init__(self):
        self._pars = []
        self._np_pars = np.array([])
        self._pars_dict = {}
        self._inverted_pars_dict = None
        self._parameter_infos = []

    def add_parameter(
            self,
            name: str,
            parameter_type: str,
            floating: bool,
            initial_value: float
    ):
        if name in self._pars_dict:
            raise ValueError(f"Trying to register new parameter with name {name} in ParameterHandler, \n"
                             f"but a parameter with this name is already registered with the following properties:\n"
                             f"\t-" + "\n\t-".join(self._parameter_infos[self._pars_dict[name]].info_as_string_list()))
        self._check_parameter_types(parameter_types=[parameter_type])

        parameter_index = len(self._pars)
        assert parameter_index not in self._pars_dict.values(), (parameter_index, self._pars_dict.values())

        parameter_info = ParameterInfo(
            index=parameter_index,
            name=name,
            parameter_type=parameter_type,
            floating=floating,
            initial_value=initial_value
        )

        self._pars.append(initial_value)
        self._np_pars = np.array(self._pars)
        assert len(self._pars) == len(self._np_pars), (len(self._pars), len(self._np_pars))
        self._pars_dict[name] = parameter_index
        self._inverted_pars_dict = None
        assert len(self._pars) == len(self._pars_dict), (len(self._pars), len(self._pars_dict))
        self._parameter_infos.append(parameter_info)
        assert len(self._pars) == len(self._parameter_infos), (len(self._pars), len(self._parameter_infos))

        return parameter_index

    def add_parameters(
            self,
            names: List[str],
            parameter_types: Union[str, List[str]],
            floating: List[bool],
            initial_values: Union[np.ndarray, List[float]]
    ) -> List[int]:
        self._check_parameters_input(
            names=names,
            parameter_types=parameter_types,
            floating=floating,
            initial_values=initial_values
        )

        if isinstance(parameter_types, str):
            self._check_parameter_types(parameter_types=[parameter_types])
            types_list = [parameter_types] * len(names)
        else:
            self._check_parameter_types(parameter_types=parameter_types)
            types_list = parameter_types

        indices = []
        for name, p_type, flt, initial_val in zip(names, types_list, floating, initial_values):
            index = self.add_parameter(
                name=name,
                parameter_type=p_type,
                floating=flt,
                initial_value=initial_val
            )
            indices.append(index)

        return indices

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

    @staticmethod
    def _check_parameters_input(
            names: List[str],
            parameter_types: Union[str, List[str]],
            floating: List[bool],
            initial_values: Union[np.ndarray, List[float]]
    ) -> None:
        if isinstance(initial_values, np.ndarray) and len(initial_values.shape) != 1:
            raise ValueError(f"Parameter 'initial_values' must be 1 dimensional, "
                             f"but has shape {initial_values.shape}...")
        if isinstance(initial_values, list):
            if not all(isinstance(value, float) for value in initial_values):
                raise ValueError(f"Parameter'initial_values' must be list of floats or 1 dimensional numpy array!")

        if not len(names) == len(initial_values):
            raise ValueError(f"The provided number of 'inital_values' (= {len(initial_values)}) and number of 'names'"
                             f" (= {len(names)}) does not match! Must be of same length!")
        if not len(floating) == len(initial_values):
            raise ValueError(f"The provided number of 'initial_values' (= {len(initial_values)}) and number of "
                             f"'floating' (= {len(floating)}) does not match! Must be of same length!")
        if not len(parameter_types) == len(initial_values) or len(parameter_types) == 1:
            raise ValueError(f"The provided number of 'initial_values' (= {len(initial_values)}) and number of "
                             f"'parameter_types' (= {len(parameter_types)}) does not match!\n"
                             f"The 'parameter_types' must be a list of same length as the other arguments,\n"
                             f"or a single string which will be used for all parameters!")
        if not len(names) == len(set(names)):
            raise ValueError(f"Entries of 'names' should each be unique! You provided:\n{names}")

    @staticmethod
    def _check_parameter_types(parameter_types: List[str]):
        for parameter_type in parameter_types:
            raise ValueError(f"Trying to add new parameter with unknown parameter_type {parameter_type}!\n"
                             f"Parameter_type must be one of {ParameterHandler.parameter_types}!")


class Parameter(ABC):
    # TODO: Maybe allow to temporarily overwrite the parameter for tests/plotting or whatever.
    def __init__(
            self,
            name: str,
            parameter_handler: ParameterHandler,
            parameter_type: str,
            floating: bool,
            initial_value: float
    ):
        if parameter_type not in ParameterHandler.parameter_types:
            raise ValueError(f"Parameter type must be one of {ParameterHandler.parameter_types}!\n"
                             f"You provided '{parameter_type}'...")
        self._name = name
        self._params = parameter_handler
        self._parameter_type = parameter_type
        self._floating = floating
        self._initial_value = initial_value

        self._index = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def parameter_type(self) -> str:
        return self._parameter_type

    @property
    def floating(self) -> bool:
        return self._floating

    @property
    def initial_value(self) -> float:
        return self._initial_value

    @property
    def value(self) -> float:
        if self._index is None:
            return self.initial_value
        return self._params.get_parameters_by_index(self._index)

    @property
    def index(self) -> Optional[int]:
        return self._index

    @index.setter
    def index(self, index: int) -> None:
        if self._index is not None:
            raise RuntimeError("Trying to reset a parameter index!")
        self._index = index

    @property
    def parameter_handler(self) -> ParameterHandler:
        return self._params

    @abstractmethod
    def _additional_info(self) -> Optional[str]:
        return None

    def as_string(self) -> str:
        output = f"{self.__class__.__name__}" \
                 f"\n\tname: {self._name}" \
                 f"\n\tparameter type: {self._parameter_type}" \
                 f"\n\tindex: {self._index}" \
                 f"\n\tfloating: {self._floating}" \
                 f"\n\tinitial value: {self._initial_value}"
        if self._additional_info is not None:
            output += self._additional_info
        return output

    def __eq__(self, other: "Parameter") -> bool:
        if not self.index == other.index:
            return False
        if not self.floating == other.floating:
            return False
        if not self.initial_value == other.initial_value:
            return False
        if self.parameter_handler is not other.parameter_handler:
            return False

        return True


class TemplateParameter(Parameter):
    def __init__(
            self,
            name: str,
            parameter_handler: ParameterHandler,
            parameter_type: str,
            floating: bool,
            initial_value: float,
            index: Optional[int]
    ):
        super().__init__(
            name=name,
            parameter_handler=parameter_handler,
            parameter_type=parameter_type,
            floating=floating,
            initial_value=initial_value
        )
        self._base_model_parameter = None

        if index is not None:
            self.index = index

    @property
    def base_model_parameter(self) -> "ModelParameter":
        return self._base_model_parameter

    @base_model_parameter.setter
    def base_model_parameter(self, base_model_parameter: "ModelParameter") -> None:
        assert self._base_model_parameter is None
        self._base_model_parameter = base_model_parameter

    def _additional_info(self) -> Optional[str]:
        return f"\n\tbase model parameter index: {self._base_model_parameter.model_index}"


class ModelParameter(Parameter):
    def __init__(
            self,
            name: str,
            parameter_handler: ParameterHandler,
            parameter_type: str,
            model_index: int,
            floating: bool,
            initial_value: float
    ):
        super().__init__(
            name=name,
            parameter_handler=parameter_handler,
            parameter_type=parameter_type,
            floating=floating,
            initial_value=initial_value
        )
        self._usage_list = []
        self._model_index = model_index

        self._register_in_parameter_handler()

    @property
    def model_index(self) -> int:
        return self._model_index

    def _register_in_parameter_handler(self) -> None:
        index = self._params.add_parameter(
            name=self.name,
            parameter_type=self.parameter_type,
            floating=self.floating,
            initial_value=self.initial_value
        )
        self.index = index

    def used_by(
            self,
            template_parameter: TemplateParameter,
            template_serial_number: int
    ):
        template_parameter.base_model_parameter = self
        info_tuple = (template_parameter, template_serial_number)
        assert not any(info_tuple == entry for entry in self._usage_list), info_tuple
        self._usage_list.append(info_tuple)

    @property
    def usage_list(self) -> List[Tuple[TemplateParameter, int]]:
        return self._usage_list

    @property
    def usage_serial_number_list(self) -> List[int]:
        return [serial_number for _, serial_number in self._usage_list]

    def _additional_info(self) -> Optional[str]:
        return f"\n\tused by templates with the serial numbers: {self.usage_serial_number_list}"
