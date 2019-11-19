"""
Provides the Histogram class and the HistogramContainer class.

The Histogram class can hold one or multiple HistogramComponents.
The Histogram defines the Binning and other features for all its components and
orchestrates the necessary calculations on all components.

The container class HistogramContainer can hold different Histograms which are
to be plotted in the same plot.
"""
import logging
import numpy as np
import matplotlib.pyplot as plt

from collections import OrderedDict
from typing import Optional, Union, List, Tuple, Dict, ItemsView, KeysView, ValuesView

from templatefitter.binned_distributions.binning import Binning
from templatefitter.binned_distributions import distributions_utility
from templatefitter.binned_distributions.binned_distribution import BinnedDistribution
from templatefitter.binned_distributions.distributions_utility import get_combined_covariance

from templatefitter.plotter import plot_style
from templatefitter.plotter.histogram_variable import HistVariable
from templatefitter.plotter.histogram_component import HistComponent

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "Histogram",
    "HistogramContainer"
]

plot_style.set_matplotlibrc_params()


class Histogram:
    """
    Class which holds several HistogramComponents,
    defines their features and performs operations on them.

    The Histogram is a stacked histogram composed of all its components.
    """
    valid_hist_types = ["bar", "barstacked", "step", "stepfilled"]
    default_hist_type = "stepfilled"

    def __init__(self, variable: HistVariable, hist_type: Optional[str] = None) -> None:
        if not isinstance(variable, HistVariable):
            raise ValueError(f"The parameter 'variable' must be a HistVariable instance, "
                             f"but you provided an object of type {type(variable).__name__}")
        self._variable = variable
        self._hist_type = self._check_and_return_hist_type(hist_type=hist_type)

        self._binning = Binning(
            bins=variable.n_bins,
            dimensions=1,
            scope=variable.scope,
            log_scale=variable.use_log_scale
        )

        self._components = []  # type: List[HistComponent]
        self._auto_color_index = 0

        self._raw_data_scope = None
        self._covariance_matrix = None
        self._binning_used_for_covariance_matrix = None

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

        if component.input_column_name is None:
            component.input_column_name = self.variable.df_label
        else:
            assert component.input_column_name == self.variable.df_label, \
                (component.input_column_name, self.variable.df_label)

        self._components.append(component)

    def add_histogram_component_via_constructor(self, *args, **kwargs):
        new_component = HistComponent(*args, **kwargs)
        self.add_histogram_component_directly(component=new_component)

    def reset_binning_to_use_raw_data_scope(self):
        new_binning = Binning(
            bins=self.binning.bin_edges,
            dimensions=self.binning.dimensions,
            scope=self.raw_data_range,
            log_scale=self.variable.use_log_scale
        )
        self._binning = new_binning

    def apply_adapted_binning(self, minimal_bin_count: int = 5, minimal_number_of_bins: int = 7):
        new_bin_edges = distributions_utility.run_adaptive_binning(
            distributions=self._get_underlying_distributions(),
            bin_edges=self.binning.bin_edges,
            minimal_bin_count=minimal_bin_count,
            minimal_number_of_bins=minimal_number_of_bins,
        )
        new_binning = Binning(
            bins=new_bin_edges,
            dimensions=self.binning.dimensions,
            scope=self.binning.range,
            log_scale=self.variable.use_log_scale
        )
        self._binning = new_binning

    def get_bin_counts(self) -> List[np.ndarray]:
        return [component.get_histogram_bin_count(binning=self.binning) for component in self._components]

    def get_bin_count_of_component(self, index: int) -> np.ndarray:
        return self._components[index].get_histogram_bin_count(binning=self.binning)

    def get_statistical_uncertainty_per_bin(self, normalization_factor: Optional[float] = None) -> np.ndarray:
        return np.sum([
            component.get_histogram_squared_bin_errors(binning=self.binning, normalization_factor=normalization_factor)
            for component in self._components
        ], axis=0)

    def get_systematic_uncertainty_per_bin(self) -> Optional[np.ndarray]:
        cov = self.get_covariance_matrix()
        if cov is not None:
            return np.diagonal(cov)
        else:
            return None

    def get_covariance_matrix(self) -> np.ndarray:
        if self._covariance_matrix is not None:
            if self._binning_used_for_covariance_matrix == self.binning:
                return self._covariance_matrix

        covariance_matrix = get_combined_covariance(
            distributions=[comp.get_underlying_binned_distribution(binning=self.binning) for comp in self._components]
        )
        self._covariance_matrix = covariance_matrix
        self._binning_used_for_covariance_matrix = self.binning
        return covariance_matrix

    def get_component(self, index: int) -> HistComponent:
        return self._components[index]

    @property
    def variable(self) -> HistVariable:
        return self._variable

    @property
    def hist_type(self) -> str:
        return self._hist_type

    @property
    def number_of_components(self) -> int:
        return len(self._components)

    @property
    def colors(self) -> Tuple[str, ...]:
        return tuple([comp.color for comp in self._components])

    @property
    def labels(self) -> Tuple[str, ...]:
        return tuple([comp.label for comp in self._components])

    @property
    def binning(self) -> Binning:
        return self._binning

    @property
    def number_of_bins(self) -> int:
        assert len(self.binning.num_bins) == 1, self.binning.num_bins
        return self.binning.num_bins[0]

    def reset_binning(self, new_binning: Binning) -> None:
        self._binning = new_binning

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

    @property
    def raw_weight_sum(self) -> float:
        return sum([component.raw_weights_sum for component in self._components])

    @property
    def raw_data_size(self) -> int:
        return sum([component.raw_data_size for component in self._components])

    @property
    def raw_data_range(self) -> Tuple[float, float]:
        return self._find_range_from_components()

    def _get_underlying_distributions(self) -> List[BinnedDistribution]:
        return [component.get_underlying_binned_distribution(binning=self.binning) for component in self._components]

    def _find_range_from_components(self) -> Tuple[float, float]:
        if self._raw_data_scope is not None:
            return self._raw_data_scope

        min_values = []
        max_values = []

        if len(self._components) == 0:
            raise RuntimeError("No components available to derive raw data range from...")

        for component in self._components:
            if component.input_column_name is None:
                component.input_column_name = self.variable.df_label
            else:
                assert component.input_column_name == self.variable.df_label, \
                    (component.label, component.input_column_name, self.variable.df_label)

            min_values.append(np.amin(component.min_val))
            max_values.append(np.amax(component.max_val))

        scope_tuple = (np.amin(min_values), np.amax(max_values))
        self._raw_data_scope = scope_tuple
        return scope_tuple

    def _check_and_return_hist_type(self, hist_type: Optional[str]) -> str:
        if hist_type is None:
            return self.default_hist_type

        base_error_text = f"Argument 'hist_type' must be either None or one of the strings " \
                          f"{self.valid_hist_types} as expected by matplotlib.pyplot.hist.\n"
        if not isinstance(hist_type, str):
            raise TypeError(f"{base_error_text}You provided an object of type {type(hist_type).__name__}!")
        if hist_type not in self.valid_hist_types:
            raise ValueError(f"{base_error_text}You provided the string {hist_type}!")

        return hist_type

    def _get_auto_color(self):
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        color = colors[self._auto_color_index % len(colors)]
        self._auto_color_index += 1
        return color


class HistogramContainer:
    """
    Container class which holds different Histograms and their plot attributes.
    The Histograms are sorted and addressable via a key string, i.e. the container
    is basically an OrderedDict.
    """

    def __init__(self):
        self._histogram_dict = OrderedDict()  # type: Dict[str, Histogram]
        self._common_binning = None  # type: Optional[Binning]
        self._common_variable = None  # type: Optional[HistVariable]

    def add_histogram(self, key: str, histogram: Histogram) -> None:
        if not isinstance(key, str):
            raise TypeError(f"The parameter 'key' must be a string, "
                            f"but you provided an object of type {type(key).__name__}.")
        if not isinstance(histogram, Histogram):
            raise TypeError(f"The parameter 'histogram' must be an instance of the Histogram class, "
                            f"but you provided an object of type {type(histogram).__name__}.")
        if key in self.histogram_keys:
            raise KeyError(f"The HistogramContainer instance already contains a Histogram under the key {key}!")

        if self._common_binning is None:
            self._common_binning = histogram.binning
        else:
            if not self._common_binning == histogram.binning:
                raise RuntimeError(f"The HistogramContainer instance already contains Histograms with a different "
                                   f"Binning than the one of the Histogram you are trying to add!\n"
                                   f"Current Binning:\n\t" + "\n\t".join(self.common_binning.as_string_list)
                                   + f"\nBinning of new Histogram:\n\t" + "\n\t".join(histogram.binning.as_string_list)
                                   + "\n\nThe Binning must be the same as the one of the other Histograms!")
        if self._common_variable is None:
            self._common_variable = histogram.variable
        else:
            if not self._common_variable == histogram.variable:
                raise RuntimeError(f"The HistogramContainer instance already contains Histograms with a different "
                                   f"HistVariable than the one of the Histogram you are trying to add!\n"
                                   f"Current HistVariable:\n\t" + "\n\t".join(self.common_variable.as_string_list)
                                   + f"\nHistVariable of new Histogram:\n\t"
                                   + "\n\t".join(histogram.variable.as_string_list)
                                   + "\n\nThe HistVariable must be the same as the one of the other Histograms!")

        self._histogram_dict.update({key: histogram})

    def update_binning(self, new_binning: Binning) -> None:
        for histogram in self.histograms:
            histogram.reset_binning(new_binning=new_binning)
        self._common_binning = new_binning

    def apply_adaptive_binning_based_on_key(
            self,
            key: str,
            minimal_bin_count: int = 5,
            minimal_number_of_bins: int = 7
    ) -> None:
        histogram_to_use = self.get_histogram_by_key(key=key)
        histogram_to_use.apply_adapted_binning(
            minimal_bin_count=minimal_bin_count,
            minimal_number_of_bins=minimal_number_of_bins
        )
        new_binning = histogram_to_use.binning
        logging.info(f"Updating binning for all components via adaptive binning "
                     f"based on the Histogram with the key '{key}' to:\n\t"
                     + "\n\t".join(new_binning.as_string_list))
        self.update_binning(new_binning=new_binning)

    def reset_binning_to_use_raw_data_range_of_key(self, key: str) -> None:
        histogram_to_use = self.get_histogram_by_key(key=key)
        histogram_to_use.reset_binning_to_use_raw_data_scope()
        new_binning = histogram_to_use.binning
        logging.info(f"Updating binning for all components to use raw data range "
                     f"of the Histogram with the key '{key}'\nNew binning:\n\t"
                     + "\n\t".join(new_binning.as_string_list))
        self.update_binning(new_binning=new_binning)

    def reset_binning_to_use_raw_data_range_of_all(self) -> None:
        raw_ranges = [hist.raw_data_range for hist in self.histograms]
        full_raw_range = (min([rr[0] for rr in raw_ranges]), max([rr[1] for rr in raw_ranges]))

        new_binning = Binning(
            bins=self._common_binning.bin_edges,
            dimensions=self._common_binning.dimensions,
            scope=full_raw_range,
            log_scale=self._common_variable.use_log_scale
        )

        self.update_binning(new_binning=new_binning)

    @property
    def histogram_keys(self) -> KeysView[str]:
        return self._histogram_dict.keys()

    @property
    def histograms(self) -> ValuesView[Histogram]:
        return self._histogram_dict.values()

    @property
    def items(self) -> ItemsView[str, Histogram]:
        return self._histogram_dict.items()

    def get_histogram_by_key(self, key: str) -> Histogram:
        return self._histogram_dict[key]

    def __getitem__(self, key: str) -> Histogram:
        return self._histogram_dict[key]

    @property
    def number_of_histograms(self) -> int:
        return len(self._histogram_dict)

    @property
    def common_binning(self) -> Optional[Binning]:
        return self._common_binning

    @property
    def common_variable(self) -> Optional[HistVariable]:
        return self._common_variable

    @property
    def common_bin_mids(self) -> np.ndarray:
        bin_mids = self.common_binning.bin_mids
        assert len(bin_mids) == 1, bin_mids
        return np.array(bin_mids[0])
