"""
Parameter Handler
    Class for managing parameters of all templates contained within a fit model.
"""
import logging
import numpy as np

from abc import ABC, abstractmethod
from typing import Optional, Union, List, Tuple, Dict, NamedTuple

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "ParameterHandler",
    "TemplateParameter",
    "ModelParameter"
]


class ParameterInfo(NamedTuple):
    index: int
    name: str
    model_index: int
    parameter_type: str
    floating: bool
    initial_value: float
    constraint_value: Optional[float]
    constraint_sigma: Optional[float]

    def info_as_string_list(self) -> List[str]:
        return [
            f"index: {self.index}",
            f"name: {self.name}",
            f"model_index: {self.model_index}",
            f"parameter_type: {self.parameter_type}",
            f"floating: {self.floating}",
            f"initial_value: {self.initial_value}",
            f"constraint_value: {self.constraint_value}",
            f"constraint_sigma: {self.constraint_sigma}",
        ]

    def as_string(self) -> str:
        info_list = self.info_as_string_list()
        return f"Parameter {info_list[0]}:\n" + f"\n\t".join(info_list)


class ParameterResetInfo(NamedTuple):
    index: int
    name: str
    new_initial_value: float
    old_initial_value: float


class ParameterHandler:
    yield_parameter_type = "yield"
    fraction_parameter_type = "fraction"
    efficiency_parameter_type = "efficiency"
    bin_nuisance_parameter_type = "bin_nuisance"
    parameter_types = [
        yield_parameter_type,
        fraction_parameter_type,
        efficiency_parameter_type,
        bin_nuisance_parameter_type
    ]

    def __init__(self):
        self._initial_pars = []  # type: List[float]
        self._np_pars = np.array([])
        self._pars_dict = {}  # type: Dict[str, int]
        self._inverted_pars_dict = None  # type: Optional[Dict[int, str]]
        self._parameter_infos = []  # type: List[ParameterInfo]
        self._parameters_by_type = {k: [] for k in ParameterHandler.parameter_types}
        self._redefined_params_dict = {}  # type: Dict[str, ParameterResetInfo]

        self._floating_mask = None
        self._floating_conversion_vector = None
        self._floating_conversion_matrix = None
        self._floating_parameter_indices = None
        self._initial_values_of_floating_parameters = None

        self._is_finalized = False

    def add_parameter(
            self,
            name: str,
            model_index: int,
            parameter_type: str,
            floating: bool,
            initial_value: float,
            constraint_value: Optional[float],
            constraint_sigma: Optional[float]
    ) -> int:
        if name in self._pars_dict:
            raise ValueError(f"Trying to register new parameter with name {name} in ParameterHandler, \n"
                             f"but a parameter with this name is already registered with the following properties:\n"
                             f"\t-" + "\n\t-".join(self._parameter_infos[self._pars_dict[name]].info_as_string_list()))
        self._check_parameter_types(parameter_types=[parameter_type])
        self._check_constraint_input(constraint_value=constraint_value, constraint_sigma=constraint_sigma)

        parameter_index = len(self._initial_pars)
        assert parameter_index not in self._pars_dict.values(), (parameter_index, self._pars_dict.values())
        assert parameter_index not in self._parameters_by_type[parameter_type], \
            (parameter_type, parameter_index, self._parameters_by_type[parameter_type], self._parameters_by_type)

        parameter_info = ParameterInfo(
            index=parameter_index,
            name=name,
            model_index=model_index,
            parameter_type=parameter_type,
            floating=floating,
            initial_value=initial_value,
            constraint_value=constraint_value,
            constraint_sigma=constraint_sigma
        )

        self._initial_pars.append(initial_value)
        self._np_pars = np.array(self._initial_pars)
        assert len(self._initial_pars) == len(self._np_pars), (len(self._initial_pars), len(self._np_pars))
        self._pars_dict[name] = parameter_index
        self._inverted_pars_dict = None
        assert len(self._initial_pars) == len(self._pars_dict), (len(self._initial_pars), len(self._pars_dict))
        self._parameter_infos.append(parameter_info)
        assert len(self._initial_pars) == len(self._parameter_infos), \
            (len(self._initial_pars), len(self._parameter_infos))
        self._parameters_by_type[parameter_type].append(parameter_index)
        assert sum(len(values) for values in self._parameters_by_type.values()) == len(self._initial_pars), \
            (sum(len(values) for values in self._parameters_by_type.values()), len(self._initial_pars))

        return parameter_index

    def add_parameters(
            self,
            names: List[str],
            model_indices: List[int],
            parameter_types: Union[str, List[str]],
            floating: List[bool],
            initial_values: Union[np.ndarray, List[float]],
            constraint_values: Optional[List[Union[float, None]]],
            constraint_sigmas: Optional[List[Union[float, None]]]
    ) -> List[int]:
        self._check_parameters_input(
            names=names,
            model_indices=model_indices,
            parameter_types=parameter_types,
            floating=floating,
            initial_values=initial_values,
            constraint_values=constraint_values,
            constraint_sigmas=constraint_sigmas
        )
        if constraint_values is None:
            assert constraint_sigmas is None, (constraint_values, constraint_sigmas)
            c_values = [None] * len(names)
            c_sigmas = [None] * len(names)
        else:
            c_values = constraint_values
            c_sigmas = constraint_sigmas

        if isinstance(parameter_types, str):
            self._check_parameter_types(parameter_types=[parameter_types])
            types_list = [parameter_types] * len(names)
        else:
            self._check_parameter_types(parameter_types=parameter_types)
            types_list = parameter_types

        indices = []
        zipped_infos = zip(names, model_indices, types_list, floating, initial_values, c_values, c_sigmas)
        for name, m_index, p_type, flt, initial_val, cv, cs in zipped_infos:
            index = self.add_parameter(
                name=name,
                model_index=m_index,
                parameter_type=p_type,
                floating=flt,
                initial_value=initial_val,
                constraint_value=cv,
                constraint_sigma=cs
            )
            indices.append(index)

        return indices

    def add_constraint_to_parameter(self, index: int, constraint_value: float, constraint_sigma: float) -> None:
        self._check_constraint_input(constraint_value=constraint_value, constraint_sigma=constraint_sigma)
        assert index not in [i for i, _, _ in self.get_constraint_information()]
        assert self._parameter_infos[index].constraint_value is None, self._parameter_infos[index].constraint_value
        assert self._parameter_infos[index].constraint_sigma is None, self._parameter_infos[index].constraint_sigma

        self._parameter_infos[index].constraint_value = constraint_value
        self._parameter_infos[index].constraint_sigma = constraint_sigma

        assert index in [i for i, _, _ in self.get_constraint_information()]

    def finalize(self) -> None:
        assert not self._is_finalized
        assert self._floating_mask is None
        assert self._floating_conversion_vector is None
        assert self._floating_conversion_matrix is None

        self._floating_mask = self._create_floating_parameter_mask()
        self._floating_conversion_vector = self._create_conversion_vector()
        self._floating_conversion_matrix = self._create_conversion_matrix()

        self._check_parameter_conversion()

        self._create_floating_parameter_indices_info()
        self._create_floating_parameter_initial_value_info()

        self._is_finalized = True

    def _create_floating_parameter_mask(self) -> Tuple[bool, ...]:
        return tuple([p_info.floating for p_info in self._parameter_infos])

    def _create_conversion_vector(self) -> np.ndarray:
        # Is the inverse of the floating_parameter_mask converted to integers times the parameters
        # and thus yields the fixed parameters.
        conversion_vector = np.array([0 if m else 1 for m in self.floating_parameter_mask])
        return conversion_vector * self._np_pars

    def _create_conversion_matrix(self) -> np.ndarray:
        n_floating = sum(self.floating_parameter_mask)
        assert isinstance(n_floating, int), n_floating
        conversion_matrix = np.zeros((n_floating, len(self.floating_parameter_mask)))

        i_index = 0
        for j_index, mask_bool in enumerate(self.floating_parameter_mask):
            if mask_bool:
                conversion_matrix[i_index, j_index] = 1
                i_index += 1

        return conversion_matrix

    def _create_floating_parameter_indices_info(self) -> None:
        self._floating_parameter_indices = np.array([
            index for index, floating in enumerate(self.floating_parameter_mask) if floating
        ])
        assert all(fpi == fpi_from_pi for fpi, fpi_from_pi in zip(
            self._floating_parameter_indices,
            [index for index, pi in enumerate(self._parameter_infos) if pi.floating]
        ))

    def _create_floating_parameter_initial_value_info(self, reset_parameter_name: Optional[str] = None) -> None:
        self._initial_values_of_floating_parameters = np.array([
            iv for iv, floating in zip(self._initial_pars, self.floating_parameter_mask) if floating
        ])

        floating_pis = [pi for pi, floating in zip(self._parameter_infos, self.floating_parameter_mask) if floating]
        assert all(((iv == p.initial_value)
                    or (reset_parameter_name is not None and p.name == reset_parameter_name)
                    or (p.name in self._redefined_params_dict.keys()))
                   for iv, p in zip(self._initial_values_of_floating_parameters, floating_pis)
                   ), \
            "\n\t - ".join([f"{iv}, {pi.initial_value}, {pi.name}"
                            for iv, pi in zip(self._initial_values_of_floating_parameters, floating_pis)
                            if not ((iv == pi.initial_value) or (pi.name == reset_parameter_name)
                                    or pi.name in self._redefined_params_dict.keys())])

    @property
    def floating_parameter_mask(self) -> Tuple[bool, ...]:
        return self._floating_mask

    @property
    def conversion_vector(self) -> np.ndarray:
        return self._floating_conversion_vector

    @property
    def conversion_matrix(self) -> np.ndarray:
        return self._floating_conversion_matrix

    def get_combined_parameters(self, parameter_vector: np.ndarray) -> np.ndarray:
        return parameter_vector @ self.conversion_matrix + self.conversion_vector

    def get_combined_parameters_by_index(
            self,
            parameter_vector: np.ndarray,
            indices: Union[int, List[int]]
    ) -> np.ndarray:
        # This getter combines floating parameters provided via the argument 'parameter_vector' and fixed parameters
        # and then yields the parameters with the indices provided via the 'indices' argument.
        return self.get_combined_parameters(parameter_vector=parameter_vector)[indices]

    def get_combined_parameters_by_slice(
            self,
            parameter_vector: np.ndarray,
            slicing: Tuple[Optional[int], Optional[int]]
    ) -> np.ndarray:
        # This getter combines floating parameters provided via the argument 'parameter_vector' and fixed parameters
        # and then yields the parameters for the slicing provided via the 'slicing' argument.
        return self.get_combined_parameters(parameter_vector=parameter_vector)[slicing[0]:slicing[1]]

    def _check_parameter_conversion(self) -> None:
        assert self._floating_mask is not None
        assert self._floating_conversion_vector is not None
        assert self._floating_conversion_matrix is not None

        assert len(self._floating_conversion_vector.shape) == 1, self._floating_conversion_vector.shape
        assert len(self._floating_conversion_matrix.shape) == 2, self._floating_conversion_matrix.shape
        assert len(self._floating_mask) == len(self._floating_conversion_vector), \
            (len(self._floating_mask), len(self._floating_conversion_vector))

        assert len(self._floating_conversion_vector) == self._floating_conversion_matrix.shape[1], \
            (len(self._floating_conversion_vector), self._floating_conversion_matrix.shape[1],
             self._floating_conversion_matrix.shape)

        n_floating = sum(self.floating_parameter_mask)
        assert n_floating == self._floating_conversion_matrix.shape[0], \
            (n_floating, self._floating_conversion_matrix.shape[0])

        assert all(entry == 0 if p_info.floating else p_info.initial_value == entry
                   for p_info, entry in zip(self._parameter_infos, self.conversion_vector)), \
            ([(pi.floating, pi.initial_value) for pi in self._parameter_infos], self.conversion_vector)

    def update_parameters(self, parameter_vector: np.ndarray) -> None:
        self._np_pars = parameter_vector @ self.conversion_matrix + self.conversion_vector

    def get_index(self, name: str) -> int:
        return self._pars_dict[name]

    def get_name(self, index: int) -> str:
        if self._inverted_pars_dict is None:
            _inverted_dict = {v: k for k, v in self._pars_dict.items()}  # type: Dict[int, str]
            assert len(_inverted_dict.keys()) == len(self._pars_dict.keys()), (len(_inverted_dict.keys()), len(self._pars_dict.keys()))
            self._inverted_pars_dict = _inverted_dict
        return self._inverted_pars_dict[index]

    def get_parameters_by_slice(self, slicing: Tuple[Optional[int], Optional[int]]) -> np.ndarray:
        return self._np_pars[slicing[0]:slicing[1]]

    def get_parameters_by_index(self, indices: Union[int, List[int]]) -> Union[np.ndarray, float]:
        return self._np_pars[indices]

    def get_parameters_by_name(self, parameter_names: Union[str, List[str]]) -> Union[np.ndarray, float]:
        if isinstance(parameter_names, list):
            indices = [self.get_index(name=name) for name in parameter_names]
        elif isinstance(parameter_names, str):
            indices = self.get_index(name=parameter_names)
        else:
            raise ValueError(f"Expecting string or list of strings for argument 'parameter_names'!\n"
                             f"You provided {parameter_names}")
        return self._np_pars[indices]

    def get_parameter_infos_by_index(self, indices: Union[int, List[int]]) -> List[ParameterInfo]:
        if isinstance(indices, list):
            return [self._parameter_infos[i] for i in indices]
        elif isinstance(indices, int):
            return [self._parameter_infos[indices]]
        else:
            raise ValueError(f"Expecting integer or list of integers for argument 'indices'!\nYou provided {indices}")

    def get_parameter_infos_by_name(
            self,
            parameter_names: Union[str, List[str]]
    ) -> Union[List[ParameterInfo], ParameterInfo]:
        if isinstance(parameter_names, list):
            return [self._parameter_infos[self.get_index(name=name)] for name in parameter_names]
        elif isinstance(parameter_names, str):
            return [self._parameter_infos[self.get_index(name=parameter_names)]]
        else:
            raise ValueError(f"Expecting string or list of strings for argument 'parameter_names'!\n"
                             f"You provided {parameter_names}")

    def get_parameter_indices_for_type(self, parameter_type: str) -> List[int]:
        if parameter_type not in ParameterHandler.parameter_types:
            raise ValueError(f"Trying to get indices for unknown parameter_type {parameter_type}!\n"
                             f"Parameter_type must be one of {ParameterHandler.parameter_types}!")
        return self._parameters_by_type[parameter_type]

    def get_parameter_names_for_type(self, parameter_type: str) -> Tuple[str, ...]:
        if parameter_type not in ParameterHandler.parameter_types:
            raise ValueError(f"Trying to get indices for unknown parameter_type {parameter_type}!\n"
                             f"Parameter_type must be one of {ParameterHandler.parameter_types}!")
        p_names = tuple([p.name for p in self._parameters_by_type[parameter_type]])
        p_names_2 = tuple([self._inverted_pars_dict[i] for i in self.get_parameter_indices_for_type(parameter_type=parameter_type)])
        assert set(p_names) == set(p_names_2), (p_names, p_names_2)
        assert len(p_names) == len(p_names_2), (p_names, p_names_2)
        return p_names

    def get_yield_parameter_names(self) -> Tuple[str, ...]:
        return self.get_parameter_names_for_type(parameter_type=self.yield_parameter_type)

    def get_parameter_dictionary(self) -> Dict[str, int]:
        return self._pars_dict

    def get_parameter_names(self) -> List[str]:
        return list(self._pars_dict.keys())

    def get_floating_parameter_names(self) -> List[str]:
        return [name for name, floating in zip(self._pars_dict.keys(), self.floating_parameter_mask) if floating]

    def get_parameters(self) -> np.ndarray:
        return self._np_pars

    def get_floating_parameters(self) -> np.ndarray:
        return self._np_pars[self._floating_parameter_indices]

    def get_initial_values_of_floating_parameters(self) -> np.ndarray:
        assert self._is_finalized
        return self._initial_values_of_floating_parameters

    def set_parameters(self, pars: np.ndarray) -> None:
        if len(pars.shape) != 1:
            raise ValueError(f"Parameter 'pars' must be 1 dimensional, but has shape {pars.shape}...")
        if len(pars) != len(self._np_pars):
            raise ValueError(f"Length of provided parameter array (= {len(pars)}) "
                             f"does not match the length of the existing parameter array (= {len(self._np_pars)})")
        self._np_pars[:] = pars

    def reset_parameters_to_initial_values(self) -> None:
        initial_values = np.array(self._initial_pars)
        assert len(initial_values) == len(self._np_pars)
        self._np_pars[:] = initial_values

    def set_parameter_initial_value(self, parameter_name: str, new_initial_value: float) -> None:
        if parameter_name not in self._pars_dict:
            raise KeyError(f"No parameter with the name '{parameter_name}' registered!")

        parameter_index = self._pars_dict[parameter_name]
        old_value = self._initial_pars[parameter_index]
        if old_value == new_initial_value:
            return

        is_floating_parameter = self.floating_parameter_mask[parameter_index]

        self._redefined_params_dict.update({parameter_name: ParameterResetInfo(
            index=parameter_index,
            name=parameter_name,
            new_initial_value=new_initial_value,
            old_initial_value=old_value
        )})

        self._np_pars[parameter_index] = new_initial_value
        self._initial_pars[parameter_index] = new_initial_value

        if is_floating_parameter:
            self._floating_conversion_vector = self._create_conversion_vector()
            self._create_floating_parameter_initial_value_info(reset_parameter_name=parameter_name)

    def reset_parameter_initial_value(self, parameter_name: str) -> None:
        if parameter_name not in self._pars_dict:
            raise KeyError(f"No parameter with the name '{parameter_name}' registered!")
        if parameter_name not in self._redefined_params_dict:
            raise KeyError(f"No reset information for parameter '{parameter_name}' available! Cannot reset parameter!")

        parameter_reset_info = self._redefined_params_dict.pop(parameter_name)
        is_floating_parameter = self.floating_parameter_mask[parameter_reset_info.index]

        self._initial_pars[parameter_reset_info.index] = parameter_reset_info.old_initial_value
        self._np_pars[parameter_reset_info.index] = parameter_reset_info.old_initial_value

        if is_floating_parameter:
            self._floating_conversion_vector = self._create_conversion_vector()
            self._create_floating_parameter_initial_value_info()

    def reset_all_parameter_initial_values(self) -> None:
        names_of_changed_parameter = [pri for pri in self._redefined_params_dict.keys()]
        for parameter_name in names_of_changed_parameter:
            self.reset_parameter_initial_value(parameter_name=parameter_name)
        assert len(self._redefined_params_dict) == 0, len(self._redefined_params_dict)

    def get_constraint_information(self) -> Tuple[List[int], List[float], List[float]]:
        constraint_param_indices = []
        constraint_values = []
        constraint_sigmas = []
        for param_info in self._parameter_infos:
            if param_info.constraint_value is not None:
                constraint_param_indices.append(param_info.index)
                constraint_values.append(param_info.constraint_value)
                constraint_sigmas.append(param_info.constraint_sigma)

        assert len(constraint_param_indices) == len(set(constraint_param_indices)), \
            (len(constraint_param_indices), len(set(constraint_param_indices)), constraint_param_indices)
        assert all(isinstance(cv, float) for cv in constraint_values), [type(cv) for cv in constraint_values]
        assert all(isinstance(cs, float) for cs in constraint_sigmas), [type(cs) for cs in constraint_sigmas]

        return constraint_param_indices, constraint_values, constraint_sigmas

    @staticmethod
    def _check_parameters_input(
            names: List[str],
            model_indices: List[int],
            parameter_types: Union[str, List[str]],
            floating: List[bool],
            initial_values: Union[np.ndarray, List[float]],
            constraint_values: Optional[List[Union[float, None]]],
            constraint_sigmas: Optional[List[Union[float, None]]]
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
        if not len(names) == len(model_indices):
            raise ValueError(f"The provided number of 'model_indices' (= {len(model_indices)}) and number of 'names'"
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

        if not (constraint_values is None or isinstance(constraint_values, list)):
            raise ValueError(f"The optional argument 'constraint_values' can either be 'None' or a list of "
                             f"floats and 'None's.\nYou provided an object of type {type(constraint_values)}")
        if not isinstance(constraint_sigmas, type(constraint_values)):
            raise ValueError(f"The parameter 'constraint_sigmas' must be of the same type as 'constraint_values'!\n"
                             f"The must either be both None, or both lists containing 'None's and floats.")
        if isinstance(constraint_values, list):
            if not len(constraint_values) == len(names):
                raise ValueError(f"If the parameter 'constraint_values' is provided as list, it must have the same "
                                 f"length as the number of provided 'names', however, you provided lists of lengths"
                                 f"{len(constraint_values)} and {len(names)}, respectively!")
            if not all((cv is None or isinstance(cv, float)) for cv in constraint_values):
                raise ValueError(f"If the parameter 'constraint_values' is provided as list, it must contain only "
                                 f"'None's or floats!\nWhat you provided contained the following types:\n"
                                 f"{[type(cv) for cv in constraint_values]}")
            if not len(constraint_values) == len(constraint_sigmas):
                raise ValueError(f"If the parameters 'constraint_values' and 'constraint_sigmas' are provided as lists,"
                                 f" the must be of same length! However, the lists you provided have the lengths:\n"
                                 f"\tlen(constraint_values) = {len(constraint_values)}\n"
                                 f"\tlen(constraint_sigmas) = {len(constraint_sigmas)}")
            if not all(type(cv) == type(cs) for cv, cs in zip(constraint_values, constraint_sigmas)):
                raise ValueError(
                    f"The types of the 'constraint_values' and 'constraint_sigmas' list elements must "
                    f"match pairwise, however, in the lists you provided the following elements do "
                    f"not match:\n\tindex: contraint_value --- constraint_sigma"
                    + "\n\t".join([
                        f"{i}: {cv} (type {type(cv)}) --- {cs} (type {type(cs)})"
                        for i, (cv, cs) in enumerate(zip(constraint_values, constraint_sigmas))
                    ])
                )

    @staticmethod
    def _check_parameter_types(parameter_types: List[str]):
        for parameter_type in parameter_types:
            if parameter_type not in ParameterHandler.parameter_types:
                raise ValueError(f"Trying to add new parameter with unknown parameter_type {parameter_type}!\n"
                                 f"Parameter_type must be one of {ParameterHandler.parameter_types}!")

    @staticmethod
    def _check_constraint_input(constraint_value: Optional[float], constraint_sigma: Optional[float]):
        if constraint_value is None:
            if constraint_sigma is not None:
                raise ValueError(f"If the parameter 'constraint_value' is None, 'constraint_sigma' must be, too."
                                 f"You provided:\n\tconstraint_value = {constraint_value}"
                                 f"\n\tconstraint_sigma = {constraint_sigma}")
        else:
            if not isinstance(constraint_value, float):
                raise ValueError(f"The parameter must be either 'None' or a float, "
                                 f"but you provided an object of type {type(constraint_value)}")
            if not isinstance(constraint_sigma, float):
                raise ValueError(f"If the parameter 'constraint_value' is defined via a float, so must be the "
                                 f"parameter 'constraint_sigma', but you provided for the latter an object of "
                                 f"type {type(constraint_sigma)}!")


class Parameter(ABC):
    def __init__(
            self,
            name: str,
            parameter_handler: ParameterHandler,
            parameter_type: str,
            floating: bool,
            initial_value: float,
            constrain_to_value: Optional[float] = None,
            constraint_sigma: Optional[float] = None
    ):
        if parameter_type not in ParameterHandler.parameter_types:
            raise ValueError(f"Parameter type must be one of {ParameterHandler.parameter_types}!\n"
                             f"You provided '{parameter_type}'...")
        self._name = name
        self._params = parameter_handler
        self._parameter_type = parameter_type
        self._floating = floating
        self._initial_value = initial_value
        self._constraint_value = constrain_to_value
        self._constraint_sigma = constraint_sigma
        if self._constraint_value is not None and (self._constraint_value != self._initial_value):
            raise ValueError(f"If a constraint is defined for a parameter, the initial value of the parameter "
                             f"should be the value the parameter is constrained to, but the input values are:\n"
                             f"\tinitial_value = {initial_value}\n\tconstrain_to_value = {constrain_to_value}")

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
    def constraint_value(self) -> Optional[float]:
        return self._constraint_value

    @property
    def constraint_sigma(self) -> Optional[float]:
        return self._constraint_sigma

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
                 f"\n\tinitial value: {self._initial_value}" \
                 f"\n\tconstraint value: {self._constraint_value}" \
                 f"\n\tconstraint sigma: {self._constraint_sigma}"
        if self._additional_info() is not None:
            output += self._additional_info()
        return output

    def __eq__(self, other: "Parameter") -> bool:
        if not self.index == other.index:
            return False
        if not self.floating == other.floating:
            return False
        if not self.initial_value == other.initial_value:
            return False
        if not self.constraint_value == other.constraint_value:
            return False
        if not self.constraint_sigma == other.constraint_sigma:
            return False
        if not self.parameter_type == other.parameter_type:
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
            index: Optional[int],
            constrain_to_value: Optional[float] = None,
            constraint_sigma: Optional[float] = None
    ):
        super().__init__(
            name=name,
            parameter_handler=parameter_handler,
            parameter_type=parameter_type,
            floating=floating,
            initial_value=initial_value,
            constrain_to_value=constrain_to_value,
            constraint_sigma=constraint_sigma
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
            initial_value: float,
            constrain_to_value: Optional[float] = None,
            constraint_sigma: Optional[float] = None
    ):
        super().__init__(
            name=name,
            parameter_handler=parameter_handler,
            parameter_type=parameter_type,
            floating=floating,
            initial_value=initial_value,
            constrain_to_value=constrain_to_value,
            constraint_sigma=constraint_sigma
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
            model_index=self.model_index,
            parameter_type=self.parameter_type,
            floating=self.floating,
            initial_value=self.initial_value,
            constraint_value=self.constraint_value,
            constraint_sigma=self.constraint_sigma
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

    @property
    def usage_template_parameter_list(self) -> List[TemplateParameter]:
        return [template_param for template_param, _ in self._usage_list]

    def _additional_info(self) -> Optional[str]:
        return f"\n\tused by templates with the serial numbers: {self.usage_serial_number_list}"
