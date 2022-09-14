"""
This package provides
    - A simple constraint data class.
    - A ConstraintContainer class, which holds all (one or multiple) Constraints to be used in the fit model.
"""

from typing import Dict, MutableSequence, Union, Iterable, overload, Callable, TypeVar, List
from abc import abstractmethod
from collections import defaultdict
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
        numbaize: bool = True,
    ):

        self.constraint_indices = constraint_indices
        self.central_value = central_value
        self.uncertainty = uncertainty
        self._numbaize = numbaize
        self._func = function
        self._is_finalized = False

    @property
    def is_simple(self):
        return len(self.constraint_indices) == 1

    @property
    def constraint_index(self):
        if self.is_simple:
            return self.constraint_indices[0]
        else:
            return self.constraint_indices

    def finalize(self, model_parameter_mapping: Dict[str, int]):

        assert not self._is_finalized
        self._is_finalized = True
        assert self._func is not None
        self._func = ConstraintCallableCreator(self._func, model_parameter_mapping, self._numbaize)

    def __call__(self, parameter_vector):
        if self._is_finalized:
            return self._func(parameter_vector)
        else:
            raise RuntimeError("Please finalize the ComplexConstraint before calling it.")


_jitspec = [("mapping", nb.types.DictType(types.unicode_type, types.int32)), ("parameter_vector", types.float32[:])]


@jitclass(_jitspec)
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


class ConstraintCallableCreator:
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


_ConstraintT = TypeVar("_ConstraintT", Constraint, ComplexConstraint)


class ConstraintContainer(MutableSequence[_ConstraintT]):
    def __init__(self):
        super().__init__()
        self._constraints = []  # type: List[_ConstraintT]
        self._constraint_parameter_indices = defaultdict(list)  # type: defaultdict[int, List[int]]

    @overload
    def __getitem__(self, i: int) -> _ConstraintT:
        ...

    @overload
    def __getitem__(self, s: slice) -> MutableSequence[_ConstraintT]:
        ...

    def __getitem__(self, index: Union[int, slice]) -> Union[_ConstraintT, MutableSequence[_ConstraintT]]:
        if isinstance(index, slice):
            raise Exception("ConstraintContainer disallows slicing as entries are not contiguous.")
        return [self._constraints[ci] for ci in self._constraint_parameter_indices[index]]

    @overload
    def __setitem__(self, index: int, item: _ConstraintT) -> None:
        ...

    @overload
    def __setitem__(self, index: slice, item: Iterable[_ConstraintT]) -> None:
        ...

    def __setitem__(self, index: Union[int, slice], item: Union[_ConstraintT, Iterable[_ConstraintT]]) -> None:
        raise Exception("ConstraintContainer does not support setting entries directly.")

    @overload
    def __delitem__(self, index: int) -> None:
        ...

    @overload
    def __delitem__(self, index: slice) -> None:
        ...

    def __delitem__(self, index: Union[int, slice]) -> None:
        raise Exception("Items can't be deleted from ConstraintContainer.")

    def __len__(self) -> int:
        return len(self._constraints)

    def __iter__(self):
        if not len(self):
            raise RuntimeError("ConstraintContainer is empty and must be filled before iteration.")
        return iter(self._constraints)

    def __repr__(self):
        return f"ConstraintContainer({', '.join(str(c) for c in self._constraints)})"

    @abstractmethod
    def append(self, value: _ConstraintT):
        pass

    def insert(self, index: int, item: _ConstraintT) -> None:
        raise Exception("ConstraintContainer does not support insertion to preserve indices.")


class SimpleConstraintContainer(ConstraintContainer):
    """
    For each index only a simple constraint should exist.
    """

    def append(self, value: Constraint):

        if value.constraint_index in self._constraint_parameter_indices:
            raise KeyError(
                f"Simple constraint with index {value.constraint_index} already exists in ConstraintContainer,"
                f" can't add another one."
            )
        else:
            self._constraints.append(value)
            self._constraint_parameter_indices[value.constraint_index].append(len(self))


class ComplexConstraintContainer(ConstraintContainer):
    """
    For each index multiple complex constraints can exist.
    """

    def append(self, value: ComplexConstraint):

        self._constraints.append(value)
        for ci in value.constraint_indices:
            self._constraint_parameter_indices[ci].append(len(self))
