"""
Utility functions for multiple BinnedDistributions.
"""

import logging
import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple, List, Sequence
from collections.abc import Sequence as CollectionsABCSequence

from templatefitter.binned_distributions.binning import Binning, BinEdgesType
from templatefitter.binned_distributions.binned_distribution import BinnedDistribution, DataColumnNamesInput

logging.getLogger(__name__).addHandler(logging.NullHandler())

DistributionContainerInputType = Union[List[BinnedDistribution], Tuple[BinnedDistribution, ...]]

__all__ = [
    "DistributionContainerInputType",
    "get_combined_covariance",
    "find_ranges_for_data",
    "find_ranges_for_distributions",
    "run_adaptive_binning"
]


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


def find_common_binning_for_distributions(distributions: Sequence[BinnedDistribution]) -> Binning:
    assert isinstance(distributions, CollectionsABCSequence), type(distributions)
    assert all(isinstance(dist, BinnedDistribution) for dist in distributions), [type(d) for d in distributions]

    # If all distributions already have the same binning, return it.
    if all(dist.binning == distributions[0].binning for dist in distributions):
        return distributions[0].binning

    common_dims = distributions[0].dimensions
    assert all(dist.dimensions == common_dims for dist in distributions), \
        ([d.dimensions for d in distributions], common_dims)

    common_log_scale_mask = distributions[0].binning.log_scale_mask
    assert all(dist.binning.log_scale_mask == common_log_scale_mask for dist in distributions), \
        ([d.binning.log_scale_mask for d in distributions], common_log_scale_mask)

    # find most general binning:
    common_ranges = find_ranges_for_distributions(distributions=distributions)
    n_bins_per_dim = {dim: [] for dim in range(common_dims)}
    for dist in distributions:
        num_bins = dist.binning.num_bins
        for dim in range(common_dims):
            n_bins_per_dim[dim].append(num_bins[dim])
    common_num_bins = tuple([max(n_bins_per_dim[dim]) for dim in range(common_dims)])
    common_binning = Binning(
        bins=common_num_bins,
        dimensions=common_dims,
        scope=common_ranges,
        log_scale=common_log_scale_mask
    )

    return common_binning


def run_adaptive_binning(
        distributions: Sequence[BinnedDistribution],
        bin_edges: Optional[BinEdgesType] = None,
        minimal_bin_count: int = 5,
        minimal_number_of_bins: Union[int, Sequence[int]] = 7
) -> BinEdgesType:
    if not minimal_bin_count > 0:
        raise ValueError(f"minimal_bin_count must be greater than 0, the value provided is {minimal_bin_count}")
    min_count = minimal_bin_count

    assert isinstance(distributions, CollectionsABCSequence), type(distributions)
    assert all(isinstance(dist, BinnedDistribution) for dist in distributions), [type(d) for d in distributions]

    common_dims = distributions[0].dimensions
    assert all(dist.dimensions == common_dims for dist in distributions), \
        ([d.dimensions for d in distributions], common_dims)

    common_log_scale_mask = distributions[0].binning.log_scale_mask
    assert all(dist.binning.log_scale_mask == common_log_scale_mask for dist in distributions), \
        ([d.binning.log_scale_mask for d in distributions], common_log_scale_mask)

    min_num_of_bins = _get_minimal_number_of_bins(minimal_number_of_bins=minimal_number_of_bins, dimensions=common_dims)

    if bin_edges is None:
        common_binning = find_common_binning_for_distributions(distributions=distributions)
        bin_edges = common_binning.bin_edges

    # Starting Condition
    if all(len(edges) < min_num_bins for edges, min_num_bins in zip(bin_edges, min_num_of_bins)):
        logging.info(f"Adaptive binning stopped via starting condition:\nNumber of bins per dim = {min_num_of_bins}")
        return bin_edges

    bins = [np.array(list(edges)) for edges in bin_edges]
    initial_hist = np.sum(
        np.array([np.histogramdd(dist.base_data.data, bins=bins, weights=dist.base_data.weights)[0]
                  for dist in distributions]),
        axis=0
    )
    assert len(initial_hist.shape) == common_dims, (initial_hist.shape, len(initial_hist.shape), common_dims)
    assert all(initial_hist.shape[i] == len(bin_edges[i]) - 1 for i in range(common_dims)), \
        (initial_hist.shape, [len(edges) for edges in bin_edges])

    # Termination condition
    if np.all(initial_hist >= min_count):
        logging.info(f"Adaptive binning was successfully terminated.")
        return bin_edges

    # Find axis for which a rebinning is most effective:
    n_too_small_per_axis = [max(np.sum(initial_hist < min_count, axis=axis)) for axis in range(len(initial_hist.shape))]
    axis_to_update = np.argmin(n_too_small_per_axis)
    assert isinstance(axis_to_update, int), type(axis_to_update)
    bin_edges_to_update = bin_edges[axis_to_update]

    # Decide from where to start the rebinning for this axis:
    max_bin_index_tuple = np.unravel_index(np.argmax(initial_hist), initial_hist.shape)
    max_bin = max_bin_index_tuple[axis_to_update]
    assert isinstance(max_bin, int), type(max_bin)
    num_bins_in_axis = initial_hist.shape[axis_to_update]

    left_threshold = 0.15
    right_threshold = 0.85
    if max_bin / num_bins_in_axis < left_threshold:
        method = "left"
    elif max_bin / num_bins_in_axis > right_threshold:
        method = "right"
    else:
        method = "max"

    # Adopt the bin edges of the chosen axis with the chosen method:
    if method == "left":
        new_edges = _update_bin_edges(
            start_from=method,
            hist=initial_hist,
            bin_edges_to_update=bin_edges_to_update,
            min_count=min_count,
            axis_to_update=axis_to_update,
            threshold=left_threshold
        )
    elif method == "right":
        new_edges = _update_bin_edges(
            start_from=method,
            hist=initial_hist,
            bin_edges_to_update=bin_edges_to_update,
            min_count=min_count,
            axis_to_update=axis_to_update,
            threshold=right_threshold
        )
    else:  # "max"-case
        left_bound, right_bound = (max_bin - 1, max_bin + 2)
        slc = [slice(None) for _ in range(len(initial_hist.shape))]
        slc[axis_to_update] = slice(max_bin - 2, max_bin + 3, None)
        middle_slices = tuple(slc)
        slc[axis_to_update] = slice(None, left_bound, None)
        left_slices = tuple(slc)
        slc[axis_to_update] = slice(right_bound, None, None)
        right_slices = tuple(slc)
        assert np.all(initial_hist[middle_slices] >= min_count)
        original_edges_mid = bin_edges[left_bound:right_bound]
        original_edges_left = bin_edges[:left_bound]
        original_edges_right = bin_edges[right_bound:]
        initial_hist_left = initial_hist[left_slices]
        initial_hist_right = initial_hist[right_slices]

        adopted_left = _update_bin_edges(
            start_from="left",
            hist=initial_hist_left,
            bin_edges_to_update=original_edges_left,
            min_count=min_count,
            axis_to_update=axis_to_update,
            threshold=left_threshold
        )
        adopted_right = _update_bin_edges(
            start_from="right",
            hist=initial_hist_right,
            bin_edges_to_update=original_edges_right,
            min_count=min_count,
            axis_to_update=axis_to_update,
            threshold=right_threshold
        )
        new_edges = tuple(np.r_[adopted_left, original_edges_mid, adopted_right])

    assert isinstance(new_edges, tuple)
    assert new_edges[0][0] == bin_edges_to_update[0][0], (new_edges[0][0], bin_edges_to_update[0][0])
    assert new_edges[-1][1] == bin_edges_to_update[-1][1], (new_edges[-1][1], bin_edges_to_update[-1][1])

    new_bin_edges = tuple([edges if axis != axis_to_update else new_edges for axis, edges in enumerate(bin_edges)])
    assert len(new_bin_edges) == common_dims, (len(new_bin_edges), common_dims)

    # Run this function iteratively with new bin-edges:
    final_bin_edges = run_adaptive_binning(
        distributions=distributions,
        bin_edges=new_bin_edges,
        minimal_bin_count=min_count,
        minimal_number_of_bins=minimal_number_of_bins
    )

    new_vs_old_edges_zip = list(zip(final_bin_edges, bin_edges))
    assert all(new_edges[0] == old_edges[0] for new_edges, old_edges in new_vs_old_edges_zip), \
        ("New vs old lower edges:\n\t" + "\n\t".join([f"{n[0]} vs {o[0]}" for n, o in new_vs_old_edges_zip]))
    assert all(new_edges[-1] == old_edges[-1] for new_edges, old_edges in new_vs_old_edges_zip), \
        ("New vs old upper edges:\n\t" + "\n\t".join([f"{n[-1]} vs {o[-1]}" for n, o in new_vs_old_edges_zip]))

    return final_bin_edges


def _update_bin_edges(
        start_from: str,
        hist: np.ndarray,
        bin_edges_to_update: Tuple[float, ...],
        min_count: int,
        axis_to_update: int,
        threshold: float
):
    num_bins_in_axis = hist.shape[axis_to_update]
    if start_from == "left":
        possible_starting_points = np.argmax(hist < min_count, axis=axis_to_update)
        starting_point = min([p for p in possible_starting_points if p > threshold * num_bins_in_axis])
        slc = [slice(None) for _ in range(len(hist.shape))]
        slc[axis_to_update] = slice(starting_point, None, None)
        offset = 1 if hist[tuple(slc)].shape[axis_to_update] % 2 == 0 else 0
        original = bin_edges_to_update[:starting_point + offset]
        adapted = bin_edges_to_update[starting_point + offset:][1::2]
        new_edges = tuple(np.r_[original, adapted])
        return new_edges
    elif start_from == "right":
        possible_starting_points = num_bins_in_axis - np.argmax(np.flip(hist) < min_count, axis=axis_to_update)
        starting_point = max([p for p in possible_starting_points if p < threshold * num_bins_in_axis])
        slc = [slice(None) for _ in range(len(hist.shape))]
        slc[axis_to_update] = slice(None, starting_point, None)
        offset = 0 if hist[tuple(slc)].shape[axis_to_update] % 2 == 0 else 1
        original = bin_edges_to_update[starting_point + offset:]
        adapted = bin_edges_to_update[:starting_point + offset][::2]
        new_edges = tuple(np.r_[adapted, original])
        return new_edges
    else:
        raise ValueError(f"Argument start_from can be either 'left' or 'right', but you provided '{start_from}'")


def _get_minimal_number_of_bins(minimal_number_of_bins: Union[int, Sequence[int]], dimensions: int) -> List[int]:
    if isinstance(minimal_number_of_bins, int):
        if minimal_number_of_bins <= 5:
            raise ValueError(f"minimal_number_of_bins must be > 5, but is {minimal_number_of_bins}")
        return [minimal_number_of_bins for _ in range(dimensions)]
    elif isinstance(minimal_number_of_bins, CollectionsABCSequence):
        if not all(isinstance(nb, int) for nb in minimal_number_of_bins):
            raise ValueError(f"Argument 'minimal_number_of_bins' must be integer or sequence of integers, but a "
                             f"sequence containing the following types was provided:\n"
                             f"{[type(nb) for nb in minimal_number_of_bins]}")
        if not len(minimal_number_of_bins) == dimensions:
            raise ValueError(f"If 'minimal_number_of_bins' is provided as sequence of integers, the sequence must"
                             f"have the same length as the number of dimensions of the provided distributions, but "
                             f"you provided a sequence of length {len(minimal_number_of_bins)}, whereas"
                             f"the distributions have {dimensions} dimensions.")
        if not all(nb > 5 for nb in minimal_number_of_bins):
            raise ValueError(f"minimal_number_of_bins must be > 5 for each dimension, "
                             f"but you provided {minimal_number_of_bins}")
        return [nb for nb in minimal_number_of_bins]
    else:
        raise ValueError(f"Expected integer or sequence of integer for argument 'minimal_number_of_bins', "
                         f"but you provided an object of type {type(minimal_number_of_bins)}.")
