"""
Provides container class holding the information about variables which shall
be
"""

import numpy as np
import pandas as pd

from math import floor, log10
from collections.abc import Sequence as ABCSequence
from typing import Union, Optional, Tuple, List, Sequence

__all__ = [
    "HistVariable",
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
        use_log_scale: bool = False,
    ):
        r"""
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
        self._df_label = df_label  # type: str
        self._n_bins = n_bins  # type: int
        self._scope = scope  # type: Optional[Tuple[float, float]]

        self._var_name = "" if var_name is None else var_name  # type: str
        self._x_label = self._var_name if unit is None else self._var_name + f" in {unit}"  # type: str

        self._unit = unit  # type: Optional[str]
        self._use_log_scale = use_log_scale  # type: bool

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
        assert self._scope is not None
        return self._scope

    @scope.setter
    def scope(self, value: Sequence[float]) -> None:
        error_text = (
            f"The scope must be a tuple or any other sequence of two floats." f"You provided an {type(value).__name__}"
        )
        if not isinstance(value, ABCSequence):
            raise ValueError(f"{error_text}.")
        if not len(value) == 2:
            raise ValueError(f"{error_text} with a length of {len(value)}.")
        if not all(isinstance(v, float) for v in value):
            raise ValueError(
                f"{error_text} containing objects of the types "
                f"{type(value[0]).__name__} and {type(value[1]).__name__}."
            )
        _scope = (value[0], value[1])  # type: Tuple[float, float]
        self._scope = _scope

    @property
    def variable_name(self) -> str:
        return self._var_name

    @property
    def x_label(self) -> str:
        r"""
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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HistVariable):
            raise TypeError(
                f"HistVariable can only be compared to another HistVariable, but an object of type {type(other).__name__} was provided."
            )
        if self.df_label != other.df_label:
            return False
        if self.variable_name != other.variable_name:
            return False
        if self.unit != other.unit:
            return False
        if self.use_log_scale != other.use_log_scale:
            return False
        if self.n_bins != other.n_bins:
            return False
        if self.scope != other.scope:
            return False
        return True

    def __hash__(self):
        return hash((self.df_label, self.variable_name, self.n_bins, self.scope, self.unit, self.use_log_scale))

    @property
    def as_string_list(self) -> List[str]:
        string_list = [
            f"df_label = {self.df_label}",
            f"variable_name = {self.variable_name}",
            f"unit = {self.unit}",
            f"use_log_scale = {self.use_log_scale}",
            f"n_bins = {self.n_bins}",
            f"scope = {self.scope}",
        ]
        return string_list

    @staticmethod
    def get_scoped_histogram_variable(
        base_hist_var: "HistVariable",
        dfs: Sequence[pd.DataFrame],
        round_scope_precision: int = 0,
        check_for_nans: bool = False,
    ) -> "HistVariable":
        assert all(base_hist_var.df_label in df.columns for df in dfs), base_hist_var.df_label

        if base_hist_var.scope is not None:
            return base_hist_var

        if any(len(df.index) == 0 for df in dfs):
            raise RuntimeError(
                f"Encountered empty data frames in get_scoped_histogram_variable: " f"{[len(df.index) for df in dfs]}"
            )

        if check_for_nans:
            assert all(not df[base_hist_var.df_label].isnull().values.any() for df in dfs), (
                base_hist_var.df_label,
                [df[base_hist_var.df_label].isnull().values.any() for df in dfs],
            )

        new_scope = HistVariable.round_scope_to_significance(
            lower=float(np.nanmin(np.concatenate([df[base_hist_var.df_label].values for df in dfs]))),
            upper=float(np.nanmax(np.concatenate([df[base_hist_var.df_label].values for df in dfs]))),
            improve_precision_by=round_scope_precision,
        )  # type: Tuple[Union[float, int], Union[float, int]]
        assert all(not np.isnan(x) for x in new_scope), (new_scope, base_hist_var.df_label)

        return HistVariable(
            df_label=base_hist_var.df_label,
            n_bins=base_hist_var.n_bins,
            scope=new_scope,
            var_name=base_hist_var.variable_name,
            unit=base_hist_var.unit,
            use_log_scale=base_hist_var.use_log_scale,
        )

    @staticmethod
    def round_scope_to_significance(
        lower: Union[float, int],
        upper: Union[float, int],
        improve_precision_by: int = 0,
    ) -> Tuple[float, float]:
        assert not np.isnan(lower), lower
        assert not np.isnan(upper), upper
        if not improve_precision_by >= 0:
            raise ValueError(
                f"Values for parameter improve_precision_by must be >= 0, " f"but {improve_precision_by} was provided!"
            )
        if improve_precision_by == 0:
            return lower, upper

        precision = improve_precision_by - int(floor(log10(abs(upper - lower))))
        assert not np.isnan(precision), precision
        correction = 0.5 * round(abs(upper - lower), precision)
        assert not np.isnan(correction), correction
        return round(lower - correction, precision), round(upper + correction, precision)
