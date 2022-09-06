"""
This package provides
    - A simple constraint data class.
    - A ConstraintContainer class, which holds all (one or multiple) Constraints to be used in the fit model.
"""

from typing import Dict, MutableSequence, Union, Iterable, overload, Callable
from dataclasses import dataclass
import numba as nb
import numpy as np

from numba.core import types
from numba.experimental import jitclass


@dataclass
class Constraint:
    """Class to keep track of simple constraints."""

    constraint_index: int
    central_value: float
    uncertainty: float


class ComplexConstraint(Constraint):
    def __init__(
        self,
        constraint_indices: Iterable[int],
        central_value: float,
        uncertainty: float,
        function: Callable,
        model_parameter_mapping: Dict[str, int],
        numbaize: bool = True,
    ):

        self.constraint_indices = constraint_indices
        self.central_value = central_value
        self.uncertainty = uncertainty

        self.func = ComplexConstraintCreator(function, model_parameter_mapping, numbaize)

    @property
    def is_simple(self):
        return len(self.constraint_indices) == 1

    @property
    def constraint_index(self):
        if self.is_simple:
            return self.constraint_indices[0]
        else:
            raise AttributeError("This constraint has multiple indices.")

    def __call__(self, parameter_vector):
        return self.func(parameter_vector)


jitspec = [("mapping", nb.types.DictType(types.unicode_type, types.int32)), ("parameter_vector", types.float32[:])]


@jitclass(jitspec)
class ParameterTranslator:
    """
    Jit-ized helper class which translates __getitem__ calls with parameter names to array indices.
    """

    def __init__(self, mapping, parameter_vector):
        self.mapping = mapping
        self.parameter_vector = parameter_vector

    def __getitem__(self, index):
        return self.parameter_vector[self.mapping[index]]

    def update_parameter_vector(self, parameter_vector):
        self.parameter_vector = parameter_vector


class ComplexConstraintCreator:
    """
    This class translates callables using parameter names to indices of the parameter mapper
    """

    def __init__(self, func: Callable, name_index_mapping: Dict[str, int], numbaize: bool = True):

        if numbaize:
            self.func = nb.njit(func)
        else:
            self.func = func

        self.mapping = nb.typed.Dict.empty(
            key_type=types.unicode_type,
            value_type=types.int32,
        )
        for k, v in name_index_mapping.items():
            self.mapping[k] = np.int32(v)

        self.translator = self.create_translator()

    def create_translator(self) -> ParameterTranslator:

        return ParameterTranslator(self.mapping, np.zeros(1, dtype=np.float32))

    def __call__(self, parameter_vector: np.ndarray) -> Callable:

        self.translator.update_parameter_vector(parameter_vector.astype(np.float32))
        return self.func(self.translator)


class ConstraintContainer(MutableSequence[Constraint]):
    def __init__(self):
        super().__init__()
        self._constraints = {}  # type: Dict

    @overload
    def __getitem__(self, i: int) -> Constraint:
        ...

    @overload
    def __getitem__(self, s: slice) -> MutableSequence[Constraint]:
        ...

    def __getitem__(self, index: Union[int, slice]) -> Union[Constraint, MutableSequence[Constraint]]:
        if isinstance(index, slice):
            raise Exception("ConstraintContainer disallows slicing as entries are not contiguous.")
        return self._constraints[index]

    @overload
    def __setitem__(self, index: int, item: Constraint) -> None:
        ...

    @overload
    def __setitem__(self, index: slice, item: Iterable[Constraint]) -> None:
        ...

    def __setitem__(self, index: Union[int, slice], item: Union[Constraint, Iterable[Constraint]]) -> None:
        raise Exception("ConstraintContainer does not support setting entries directly.")

    @overload
    def __delitem__(self, index: int) -> None:
        ...

    @overload
    def __delitem__(self, index: slice) -> None:
        ...

    def __delitem__(self, index: Union[int, slice]) -> None:
        if isinstance(index, slice):
            raise Exception("ConstraintContainer disallows slicing as entries are not contiguous.")
        del self._constraints[index]

    def __len__(self) -> int:
        return len(self._constraints)

    def __iter__(self):
        if not len(self._constraints):
            raise RuntimeError("ConstraintContainer is empty and must be filled before iteration.")
        return iter(dict(sorted(self._constraints.items())).values())

    def __repr__(self):
        return f"ConstraintContainer({', '.join(str(c) for c in self._constraints.values())})"

    def append(self, value: Constraint):

        if isinstance(value, ComplexConstraint):
            constraint_indices = value.constraint_indices
        elif isinstance(value, Constraint):
            constraint_indices = [value.constraint_index]
        else:
            raise TypeError(f"Argument 'value' must be of type Constraint, not of type {type(value)}.")

        for index in constraint_indices:
            if index in self._constraints:
                raise KeyError(f"Constraint with index {value.constraint_index} already exists in ConstraintContainer.")
            self._constraints[value.constraint_index] = value

    def insert(self, index: int, item: Constraint) -> None:
        raise Exception("ConstraintContainer does not support insertion to preserve indices.")
