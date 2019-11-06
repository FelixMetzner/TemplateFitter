"""
Utility functions for multiple BinnedDistributions.
"""

import numpy as np
from typing import Union, Tuple, List

from templatefitter.binned_distributions.binned_distribution import BinnedDistribution

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

# TODO: This method should be available to find the range of a distribution for a given data set
#       especially for multiple-component- or multi-dimensional distributions
# def _find_range_from_components(self) -> Tuple[float, float]:
#     min_vals = list()
#     max_vals = list()
#
#     for component in itertools.chain(*self._mc_components.values()):
#         min_vals.append(np.amin(component.data))
#         max_vals.append(np.amax(component.data))
#
#     return np.amin(min_vals), np.amax(max_vals)
