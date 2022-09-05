"""
This package provides
    - A simple constraint data class.
    - A ConstraintContainer class, which holds all (one or multiple) Constraints to be used in the fit model.
"""

from typing import NamedTuple, Dict, MutableSequence, Union, Iterable, overload


class Constraint(NamedTuple):
    """Class to keep track of simple constraints."""

    constraint_index: int
    central_value: float
    uncertainty: float


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
        if value.constraint_index in self._constraints:
            raise KeyError(f"Constraint with index {value.constraint_index} already exists in ConstraintContainer.")
        self._constraints[value.constraint_index] = value

    def insert(self, index: int, item: Constraint) -> None:
        raise Exception("ConstraintContainer does not support insertion to preserve indices.")
