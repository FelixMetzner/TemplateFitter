"""
This package provides
    - A simple constraint data class.
    - A ConstraintContainer class, which holds all (one or multiple) Constraints to be used in the fit model.
"""

from typing import Dict, MutableSequence, Union, Iterable, overload, Callable, TypeVar, List, Iterator, Optional, Sequence
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
import numba as nb
import numpy as np
import marshal
from types import LambdaType

from numba.core import types as nb_types
from numba.experimental import jitclass


@dataclass
class Constraint:
    constraint_index: int
    central_value: float
    uncertainty: float


class ComplexConstraint(Constraint):
    def __init__(
        self,
        constraint_indices: Sequence[int],
        central_value: float,
        uncertainty: float,
        function: Callable,
    ):

        self.constraint_indices = constraint_indices
        self.central_value = central_value
        self.uncertainty = uncertainty
        self._func = function  # type: Callable
        self._is_finalized = False
        self._constraint_func = None  # type: Optional[ConstraintCallableCreator]
        self._funcstring = None  # Optional[str]

        self._model_parameter_mapping = None  # type: Optional[Dict[str, int]]
        self._use_numba = False

    @property
    def is_simple(self):
        return len(self.constraint_indices) == 1

    @property
    def constraint_index(self):
        if self.is_simple:
            return self.constraint_indices[0]
        else:
            return self.constraint_indices

    def finalize(self, model_parameter_mapping: Dict[str, int], use_numba: bool):

        assert not self._is_finalized
        self._is_finalized = True
        assert self._func is not None
        assert self._constraint_func is None
        assert self._model_parameter_mapping is None

        self._model_parameter_mapping = model_parameter_mapping
        self._use_numba = use_numba

        self._constraint_func = ConstraintCallableCreator(self._func, model_parameter_mapping, use_numba=use_numba)

    def prepare_for_pickling(self):

        assert isinstance(self._func, LambdaType), "The constraint function can't be pickled, pass a lambda function."
        self._funcstring = marshal.dumps(self._func.__code__)
        del self._func
        self._constraint_func = None

    def restore_after_pickling(self):

        if self._funcstring is None:
            raise RuntimeError("Nothing to restore!")

        assert self._model_parameter_mapping is not None

        self._func = LambdaType(marshal.loads(self._funcstring), globals())
        self._constraint_func = ConstraintCallableCreator(
            self._func, self._model_parameter_mapping, use_numba=self._use_numba
        )

    def __call__(self, parameter_vector):
        if self._is_finalized:
            return self._constraint_func(parameter_vector)
        else:
            raise RuntimeError("Please finalize the ComplexConstraint before calling it.")


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

    # We save this to allow resetting the class in case of dynamically added attributes (which numba doesn't like)
    _allowed_attributes = list(ParameterTranslator.__dict__.keys())

    def __init__(self, func: Callable, name_index_mapping: Dict[str, int], use_numba: bool = True):

        self.use_numba = use_numba
        if use_numba:
            self._func = nb.njit(func)
            self._mapping = nb.typed.Dict.empty(
                key_type=nb_types.unicode_type,
                value_type=nb_types.int32,
            )
        else:
            self._func = func
            self._mapping = {}

        for k, v in name_index_mapping.items():
            self._mapping[k] = np.int32(v)

        self._translator = self._create_translator()

    def _cleanup_parameter_translator(self):
        for attribute in list(ParameterTranslator.__dict__):
            if attribute not in self._allowed_attributes:
                delattr(ParameterTranslator, attribute)

    def _create_translator(self) -> ParameterTranslator:

        if self.use_numba:
            self._cleanup_parameter_translator()

            jitspec = [
                ("mapping", nb_types.DictType(nb_types.unicode_type, nb_types.int32)),
                ("parameter_vector", nb_types.float32[:]),
            ]
            return jitclass(jitspec)(ParameterTranslator)(self._mapping, np.zeros(1, dtype=np.float32))
        else:
            return ParameterTranslator(self._mapping, np.zeros(1, dtype=np.float32))

    def __call__(self, parameter_vector: np.ndarray) -> Callable:

        self._translator.update_parameter_vector(parameter_vector.astype(np.float32))
        return self._func(self._translator)


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

    def __iter__(self) -> Iterator[_ConstraintT]:
        if not len(self._constraints):
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
