"""
Utility functions for multiple BinnedDistributions.
"""

import numpy as np
from typing import Optional


### TODO: This whole file is WIP!!! ###

# TODO: Secondly introduce utility function, which combines the systematics of multiple components of a
#       distribution and get covariance matrix and correlation matrix for this
# TODO: Maybe pack all the functions acting on multiple BinnedDistribution instances with the same binning
#       into a new file called distribution_utils.py, or so.

def _get_cov_from_systematics(self, component_label: Optional[str] = None) -> Optional[np.ndarray]:
    if component_label is not None:
        assert component_label in [c.label for c in self._mc_components["single"]]
        components = [c for c in self._mc_components["single"] if c.label == component_label]
        assert len(components) == 1
        comp = components[0]
        if comp.systematics is None:
            return None

        cov = np.zeros((len(self._bin_mids), len(self._bin_mids)))
        for sys_info in comp.systematics:
            cov += sys_info.get_cov(data=comp.data, weights=comp.weights, bin_edges=self.bin_edges())
        return cov
    else:
        components = self._mc_components["stacked"]
        if all(comp.systematics is None for comp in components):
            return None
        if all(len(comp.systematics) == 0 for comp in components):
            return None

        assert all(len(comp.systematics) == len(components[0].systematics) for comp in components)

        cov = np.zeros((len(self._bin_mids), len(self._bin_mids)))
        for sys_index in range(len(components[0].systematics)):
            assert all(isinstance(comp.systematics[sys_index], type(components[0].systematics[sys_index]))
                       for comp in components)

            varied_hists = None
            for comp in components:
                varied_hists = comp.systematics[sys_index].get_varied_hist(
                    initial_varied_hists=varied_hists,
                    data=comp.data,
                    weights=comp.weights,
                    bin_edges=self.bin_edges
                )

            cov += components[0].systematics[sys_index].get_cov_from_varied_hists(varied_hists=varied_hists)

        return cov

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
