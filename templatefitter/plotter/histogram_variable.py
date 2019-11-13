"""
Provides container class holding the information about variables which shall
be
"""

from collections.abc import Sequence as ABCSequence
from typing import Tuple, Optional, Sequence

__all__ = [
    "HistVariable"
]


class HistVariable:
    """
    Class holding the properties describing the variable which will be plotted
    with HistogramPlot classes.
    """

    def __init__(
            self,
            df_label: str,
            n_bins: int,
            scope: Optional[Tuple[float, float]] = None,
            var_name: Optional[str] = None,
            unit: Optional[str] = None,
            use_log_scale: bool = False
    ):
        """
        HistVariable constructor.
        :param df_label: Label of the variable for the column in a pandas.DataFrame.
        :param n_bins: Number of bins used in the histogram.
        :param scope: Tuple with the scope of the variable
        :param var_name: Name of the variable used for the x axis in Plots.
                         Preferably using Latex strings like r"$\mathrm{m}_{\mu\mu}$".
        :param unit: Unit of the variable, like GeV.
        :param use_log_scale: If true, x axis will be plotted in log-space.
                              Default is False.
        """
        self._df_label = df_label
        self._scope = scope
        self._var_name = var_name
        self._x_label = var_name + f' in {unit}' if unit else var_name
        self._unit = unit
        self._n_bins = n_bins
        self._use_log_scale = use_log_scale

    @property
    def df_label(self) -> str:
        """
        Column name of the variable in a pandas.DataFrame.
        :return: str
        """
        return self._df_label

    def has_scope(self) -> bool:
        """
        Checks if scope is set.
        :return: True if HistVariable has scope parameter set, False otherwise.
        """
        if self._scope is not None:
            return True
        else:
            return False

    @property
    def n_bins(self) -> int:
        """
        Number of bins used in the histogram.
        :return: int
        """
        return self._n_bins

    @property
    def scope(self) -> Tuple[float, float]:
        """
        The scope of the variable as (low, high).
        :return: Tuple[float, float]
        """
        return self._scope

    @scope.setter
    def scope(self, value: Sequence[float]) -> None:
        error_text = f"The scope must be a tuple or any other sequence of two floats." \
                     f"You provided an {type(value).__name__}"
        if not isinstance(value, ABCSequence):
            raise ValueError(f"{error_text}.")
        if not len(value) == 2:
            raise ValueError(f"{error_text} with a length of {len(value)}.")
        if not all(isinstance(v, float) for v in value):
            raise ValueError(f"{error_text} containing objects of the types "
                             f"{type(value[0]).__name__} and {type(value[1]).__name__}.")

        self._scope = tuple(value)

    @property
    def x_label(self) -> str:
        """
        X label of the variable shown in the plot.
        Accepts raw string, e.g. r"$\cos(\theta_v)$", to be interpreted as latex code.
        :return: str
        """
        if self._x_label is not None:
            return self._x_label
        else:
            return ""

    @x_label.setter
    def x_label(self, label: str) -> None:
        if not isinstance(label, str):
            raise ValueError(f"label must be a string, but an object of type {type(label).__name__}.")
        self._x_label = label

    @property
    def unit(self) -> str:
        """
        Physical unit of the variable, like "GeV".
        :return: str
        """
        if self._unit is not None:
            return self._unit
        else:
            return ""

    @unit.setter
    def unit(self, unit: str) -> None:
        if not isinstance(unit, str):
            raise ValueError(f"unit must be a string, but an object of type {type(unit).__name__}.")
        self._unit = unit

    @property
    def use_log_scale(self) -> bool:
        """
        Flag for log-scale on this axis
        :return: bool
        """
        return self._use_log_scale
