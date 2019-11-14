"""
Provides the Histogram class which will hold one or multiple HistogramComponents.
The Histogram defines the Binning and other features for all its components and
orchestrates the necessary calculations on all components.
"""
import logging
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Union

from templatefitter.binned_distributions.binning import Binning

from templatefitter.plotter import plot_style
from templatefitter.plotter.histogram_variable import HistVariable
from templatefitter.plotter.histogram_component import HistComponent

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "Histogram"
]

plot_style.set_matplotlibrc_params()


class Histogram:
    """
    Class which holds several HistogramComponents,
    defines their features and performs operations on them.

    The Histogram is a stacked histogram composed of all its components.
    """

    def __init__(self, variable: HistVariable) -> None:
        if not isinstance(variable, HistVariable):
            raise ValueError(f"The parameter 'variable' must be a HistVariable instance, "
                             f"but you provided an object of type {type(variable).__name__}")
        self._variable = variable

        self._binning = Binning(
            bins=variable.n_bins,
            dimensions=1,
            scope=variable.scope,
            log_scale=variable.use_log_scale,
        )

        self._components = []  # type: List[HistComponent]
        self._auto_color_index = 0

    def add_histogram_component(self, *args, **kwargs) -> None:
        try:
            self.add_histogram_component_via_constructor(*args, **kwargs)
        except TypeError:
            try:
                self.add_histogram_component_directly(*args, **kwargs)
            except TypeError:
                raise TypeError(f"Failed to add HistComponent. A component can be added to a Histogram\n"
                                f"\t1. by calling 'add_histogram_component' with the same signature as "
                                f"the HistComponent constructor, or"
                                f"\t2. by directly providing a HistComponent instance.\n"
                                f"You can also use the functions 'add_histogram_component_via_constructor' "
                                f"and 'add_histogram_component_directly' directly for the respective cases!\n"
                                f"You provided the following input, which did not match any of the two signatures:\n"
                                f"Positional arguments:\n\t"
                                + "\n\t".join([f"{a} of type {type(a).__name__}" for a in args])
                                + "\nKeyword arguments:\n\t"
                                + "\n\t".join([f"{k}: {v} of type {type(v).__name__}" for k, v in kwargs.items()]))

    def add_histogram_component_directly(self, component: HistComponent):
        if component.color is None:
            component.color = self._get_auto_color()
        self._components.append(component)

    def add_histogram_component_via_constructor(self, *args, **kwargs):
        new_component = HistComponent(*args, **kwargs)
        self.add_histogram_component_directly(component=new_component)

    def _find_range_from_components(self):
        # TODO
        pass

    def _get_adapted_binning(self):
        # TODO
        pass

    def get_bin_counts(self) -> List[np.ndarray]:
        return [component.get_histogram_bin_count(binning=self.binning) for component in self._components]

    @property
    def variable(self) -> HistVariable:
        return self._variable

    @property
    def binning(self) -> Binning:
        return self._binning

    @property
    def bin_edges(self) -> np.ndarray:
        bin_edges = self.binning.bin_edges
        assert len(bin_edges) == 1, (len(bin_edges), bin_edges)
        return np.array(bin_edges[0])

    @property
    def bin_mids(self) -> np.ndarray:
        bin_mids = self.binning.bin_mids
        assert len(bin_mids) == 1, (len(bin_mids), bin_mids)
        return np.array(bin_mids[0])

    @property
    def bin_widths(self) -> Union[float, np.ndarray]:
        bin_widths = self.binning.bin_widths
        assert len(bin_widths) == 1, (len(bin_widths), bin_widths)
        if all(bw == bin_widths[0][0] for bw in bin_widths[0]):
            return bin_widths[0][0]
        else:
            return np.array(bin_widths[0])

    def _get_auto_color(self):
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        color = colors[self._auto_color_index % len(colors)]
        self._auto_color_index += 1
        return color
