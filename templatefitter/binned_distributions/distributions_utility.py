"""
Utility functions for multiple BinnedDistributions.
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, List, Sequence
from collections.abc import Sequence as CollectionsABCSequence

from templatefitter.binned_distributions.binned_distribution import BinnedDistribution, DataColumnNamesInput

DistributionContainerInputType = Union[List[BinnedDistribution], Tuple[BinnedDistribution, ...]]

__all__ = ["get_combined_covariance", "DistributionContainerInputType"]


# TODO: Use this in FitModel._initialize_template_bin_uncertainties?
def get_combined_covariance(distributions: DistributionContainerInputType) -> np.ndarray:
    _check_distribution_container_input(distributions=distributions)
    common_binning = distributions[0].binning

    assert all(len(dist.systematics) == len(distributions[0].systematics) for dist in distributions), \
        ([len(d.systematics) for d in distributions])

    if all(len(dist.systematics) == 0 for dist in distributions):
        return np.eye(common_binning.num_bins_total)

    cov = np.zeros((common_binning.num_bins_total, common_binning.num_bins_total))

    if len(distributions) == 1:
        for sys_info in distributions[0].systematics:
            cov += sys_info.get_covariance_matrix(
                data=distributions[0].base_data.data,
                weights=distributions[0].base_data.weights,
                binning=common_binning
            )
        return cov
    else:
        for sys_index in range(len(distributions[0].systematics)):
            assert all(isinstance(dist.systematics[sys_index], type(distributions[0].systematics[sys_index]))
                       for dist in distributions)

            varied_hists = None
            for dist in distributions:
                varied_hists = dist.systematics[sys_index].get_varied_hist(
                    initial_varied_hists=varied_hists,
                    data=dist.base_data.data,
                    weights=dist.base_data.weights,
                    binning=common_binning
                )

            cov += distributions[0].systematics[sys_index].get_cov_from_varied_hists(varied_hists=varied_hists)

        return cov


def _check_distribution_container_input(distributions: DistributionContainerInputType) -> None:
    if not (isinstance(distributions, list) or isinstance(distributions, tuple)):
        raise ValueError(f"The argument 'distributions' must be either a list or a tuple of "
                         f"{BinnedDistribution.__name__} objects, but you provided an object of type "
                         f"{type(distributions)}...")
    if not all(isinstance(dist, BinnedDistribution) for dist in distributions):
        raise ValueError(f"The argument 'distributions' must be either a list or a tuple of "
                         f"{BinnedDistribution.__name__} objects, but you provided a list containing objects of "
                         f"the following types:\n{[type(d) for d in distributions]}")
    if not all(dist.binning == distributions[0].binning for dist in distributions):
        raise ValueError(f"The binning of all distributions provided via the argument 'distributions' must be"
                         f"the equal, however, the provided distributions had different binning definitions!")


def find_ranges_for_data(
        data: Union[np.ndarray, pd.DataFrame, pd.Series, Sequence[np.ndarray], Sequence[pd.Series]],
        data_column_names: DataColumnNamesInput = None
) -> Tuple[Tuple[Tuple[float, float], ...], int, int]:
    """
    Returns tuple of scopes and two integers representing the number dimensions of the data and the number of entries;
    the first is the desired information, the latter two are for cross-checks
    """
    if isinstance(data, pd.DataFrame):
        assert data_column_names is not None, "Column names are required if data is is provided as pandas.DataFrame!"
        if isinstance(data_column_names, List):
            assert all(isinstance(c, str) for c in data_column_names), [type(c) for c in data_column_names]
            columns = data_column_names
        else:
            assert isinstance(data_column_names, str), type(data_column_names)
            columns = [data_column_names]
        data_array = data[columns].values
    elif isinstance(data, np.ndarray):
        data_array = data
    elif isinstance(data, pd.Series):
        data_array = data.values
    elif isinstance(data, CollectionsABCSequence):
        first_type = type(data[0])
        assert all(isinstance(d_in, first_type) for d_in in data), [type(d) for d in data]
        if all(isinstance(d_in, pd.Series) for d_in in data):
            assert all(len(d_in.index) == len(data[0].index) for d_in in data), [len(d) for d in data]
            data_array = np.stack([d_in.values for d_in in data]).T
        elif all(isinstance(d_in, np.ndarray) for d_in in data):
            assert all(len(d_in.shape) == 1 for d_in in data), [d_in.shape for d_in in data]
            assert all(d_in.shape == data[0].shape for d_in in data), [d_in.shape for d_in in data]
            data_array = np.stack([d_in for d_in in data]).T
        else:
            raise ValueError(f"Unexpected input type for argument 'data': {type(data).__name__} of {first_type}...")
    else:
        raise ValueError(f"Unexpected input type for argument 'data': {type(data)}...")

    assert isinstance(data_array, np.ndarray), type(data_array)

    range_mins = data_array.min(axis=0)
    range_maxs = data_array.max(axis=0)
    range_scopes = tuple([(mi, ma) for mi, ma in zip(range_mins, range_maxs)])

    n_entries = data_array.shape[0]
    dimensions = data_array.shape[1]

    return range_scopes, dimensions, n_entries


def find_ranges_for_distributions(distributions: Sequence[BinnedDistribution]) -> Tuple[Tuple[float, float], ...]:
    common_dim = distributions[0].binning.dimensions
    assert all(dist.binning.dimensions == common_dim for dist in distributions), \
        [dist.binning.dimensions for dist in distributions]
    range_mins = {dim: [] for dim in range(common_dim)}
    range_maxs = {dim: [] for dim in range(common_dim)}

    for dist in distributions:
        for dim, range_tuple in enumerate(dist.range):
            range_mins[dim].append(range_tuple[0])
            range_maxs[dim].append(range_tuple[1])

    return tuple([(min(range_mins[dim]), max(range_maxs[dim])) for dim in range(common_dim)])
