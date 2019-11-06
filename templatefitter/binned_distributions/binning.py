"""
Class providing general methods for binning in arbitrary dimensions.
"""

import logging
import numpy as np
from typing import Union, Tuple

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["Binning", "BinEdgesType", "BinsInputType", "ScopeInputType"]

BinsInputType = Union[int, Tuple[int, ...], Tuple[float, ...], Tuple[Tuple[float, ...], ...]]
ScopeInputType = Union[None, Tuple[float, float], Tuple[Tuple[float, float], ...]]
BinEdgesType = Tuple[Tuple[float, ...]]


# TODO: Check TODO in apply_adaptive_binning!
class Binning:
    def __init__(
            self,
            bins: BinsInputType,
            dimensions: int,
            scope: ScopeInputType = None
    ) -> None:
        assert isinstance(dimensions, int) and dimensions > 0, \
            f"Dimensions must be integer greater than 0, " \
            f"you provided {dimensions} of type {type(dimensions)}!"

        self._dimensions = dimensions
        self._num_bins = None
        self._bin_edges = None
        self._bin_mids = None
        self._bin_widths = None
        self._range = None

        self._init_binning(bins_input=bins, scope_input=scope)
        self._check_binning()

    def _init_binning(
            self,
            bins_input: BinsInputType,
            scope_input: ScopeInputType = None
    ) -> None:
        error_txt = f"Ill defined binning for {self.dimensions} dimensions:\n" \
                    f"bins = {bins_input}\ntype(bins) = {type(bins_input)}"
        num_error_txt = f"If binning is defined via number of bins, scope is required, too!\n" \
                        f"bins = {bins_input}\n" \
                        f"type(bins) = {type(bins_input)}"
        scope_error_txt = f"The provided bin edges and range do not match!\n" \
                          f"Bin edges: {bins_input}\nRange: {scope_input}"

        if self.dimensions == 1:
            if isinstance(bins_input, int):
                if scope_input is None:
                    raise ValueError(num_error_txt)
                assert isinstance(scope_input, tuple) and len(scope_input) == 2, (type(scope_input), scope_input)
                assert all(isinstance(scp, int) for scp in scope_input), scope_input
                self._num_bins = (bins_input,)
                self._bin_edges = (tuple(np.linspace(*scope_input, bins_input + 1)),)
            elif isinstance(bins_input, tuple) and all(isinstance(bin_num, float) for bin_num in bins_input):
                self._num_bins = (len(bins_input) - 1,)
                self._bin_edges = (bins_input,)
                if scope_input is not None:
                    assert isinstance(scope_input, tuple) and len(scope_input) == 2, (type(scope_input), scope_input)
                    assert all(isinstance(scp, int) for scp in scope_input), scope_input
                    assert scope_input[0] == bins_input[0], scope_error_txt
                    assert scope_input[1] == bins_input[-1], scope_error_txt
            else:
                raise ValueError(error_txt)
        else:
            if not isinstance(bins_input, tuple) or self.dimensions != len(bins_input):
                raise ValueError(error_txt)
            if all(isinstance(bin_num, int) for bin_num in bins_input):
                if scope_input is None or len(bins_input) != len(scope_input) or not isinstance(scope_input, tuple):
                    raise ValueError(num_error_txt)
                assert all(isinstance(scp, tuple) and len(scp) == 2 for scp in scope_input), scope_input
                self._num_bins = bins_input
                self._bin_edges = (tuple(np.linspace(*scp, num + 1)) for num, scp in zip(bins_input, scope_input))
            elif all(isinstance(bin_num, tuple) for bin_num in bins_input):
                assert all(isinstance(edge, float) for edges in bins_input for edge in edges), bins_input
                if scope_input is not None:
                    # Just checking if provided scope fits the provided edges, not using it in this case...
                    assert isinstance(scope_input, tuple), (type(scope_input), scope_input)
                    assert all(isinstance(scp, tuple) and len(scp) == 2 for scp in scope_input), scope_input
                    assert all(scp[0] == edges[0] for scp, edges in zip(scope_input, bins_input)), scope_error_txt
                    assert all(scp[1] == edges[-1] for scp, edges in zip(scope_input, bins_input)), scope_error_txt
                self._num_bins = (len(edges) - 1 for edges in bins_input)
                self._bin_edges = bins_input
            else:
                raise ValueError(error_txt)

        self._bin_mids = tuple(map(self._get_bin_mids, self.bin_edges))
        self._bin_widths = tuple(map(self._get_bin_widths, self.bin_edges))
        self._range = tuple(map(self._get_range, self.bin_edges))

    def __eq__(self, other: "Binning") -> bool:
        if not self.dimensions == other.dimensions:
            return False
        if not self.num_bins == other.num_bins:
            return False
        if not self.bin_edges == other.bin_edges:
            return False
        if not self.range == other.range:
            return False

        return True

    def _check_binning(self) -> None:
        assert self._num_bins is None, "Number of bins is not defined after initialization!"
        assert self._bin_edges is None, "Bin edges are not defined after initialization!"

        assert isinstance(self._bin_edges, tuple)
        assert len(self._bin_edges) == self.dimensions, (len(self._bin_edges), self.dimensions)

        assert isinstance(self._num_bins, tuple), self._num_bins
        assert all(isinstance(n, int) for n in self._num_bins), self._num_bins
        assert len(self._num_bins) == self.dimensions, (len(self._num_bins), self.dimensions)
        assert all(len(self._bin_edges[i]) == bins for i, bins in enumerate(self._num_bins)), \
            (self._num_bins, self._bin_edges)

        assert len(self._bin_mids) == self.dimensions, (len(self._bin_mids), self.dimensions)
        assert all(len(m) == len(b) for m, b in zip(self._bin_mids, self._num_bins))
        assert len(self._bin_widths) == self.dimensions, (len(self._bin_widths), self.dimensions)
        assert all(len(w) == len(b) for w, b in zip(self._bin_widths, self._num_bins))
        assert len(self._range) == self.dimensions, (len(self._range), self.dimensions)
        assert all(len(r) == 2 for r in self._range)

    @staticmethod
    def _get_bin_mids(bin_edges) -> Tuple[float, ...]:
        return tuple((np.array(bin_edges)[1:] + np.array(bin_edges)[:-1]) / 2.)

    @staticmethod
    def _get_bin_widths(bin_edges) -> Tuple[float, ...]:
        return tuple(np.array(bin_edges)[1:] - np.array(bin_edges)[:-1])

    @staticmethod
    def _get_range(bin_edges) -> Tuple[float, float]:
        return bin_edges[0], bin_edges[-1]

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def num_bins(self) -> Tuple[int, ...]:
        return self._num_bins

    @property
    def num_bins_total(self) -> int:
        return sum(self._num_bins)

    @property
    def bin_edges(self) -> BinEdgesType:
        return self._bin_edges

    @property
    def bin_edges_flattened(self) -> np.ndarray:
        return np.array([edge for edges in self._bin_edges for edge in edges])

    @property
    def bin_mids(self) -> Tuple[Tuple[float, ...]]:
        return self._bin_mids

    @property
    def bin_widths(self) -> Tuple[Tuple[float, ...]]:
        return self._bin_widths

    @property
    def range(self) -> Tuple[Tuple[float, float]]:
        return self._range

    # TODO: The more complicated part is the adaptive binning of multidimensional distributions, as it is not
    #       clear into which direction the bins should be enlarged. This should be covered first, when reworking
    #       this function!
    # TODO: Rework this to be applicable to a case where multiple binned_distribution make up one histogram,
    #       maybe also in multiple channels (although multiple channels can have different binnings, maybe it
    #       could be handy to be able to find one common binning for them all... this could be done by just
    #       using the most coarse binning, whilst ensuring that this binning covers all sparsely populated
    #       regions of every other channel, too)
    def apply_adaptive_binning(
            self,
            components,  # TODO add type hint, once component class is defined
            bin_edges: np.ndarray = None,
            start_from: str = "auto",
            min_count: int = 5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.dimensions != 1:
            raise NotImplementedError("Adaptive binning is only available for 1 dimensional distributions!")

        if not min_count > 0:
            raise ValueError(f"min_count must be greater than 0, the value provided is {min_count}")

        valid_start_froms = ["left", "right", "max", "auto"]
        if start_from not in valid_start_froms:
            raise ValueError(f"Value provided for parameter `start_from` is not valid.\n"
                             f"You provided '{start_from}'.\nShould be one of {valid_start_froms}.")

        if bin_edges is None:
            bin_edges = np.array(self.bin_edges)
        bin_mids = np.array(map(self._get_bin_mids, bin_edges))
        bin_widths = np.array(map(self._get_bin_widths, bin_edges))

        # Starting Condition
        if any(len(edges) < 7 for edges in bin_edges):
            return bin_edges, bin_mids, bin_widths

        initial_hist = np.sum(
            np.array([np.histogram(comp.data, bins=bin_edges, weights=comp.weights)[0] for comp in components]),
            axis=0
        )

        # Termination condition
        if np.all(initial_hist >= min_count):
            return bin_edges, bin_mids, bin_widths

        if start_from == "left":
            starting_point = np.argmax(initial_hist < min_count)
            offset = 1 if len(initial_hist[starting_point:]) % 2 == 0 else 0
            original = bin_edges[:starting_point + offset]
            adapted = bin_edges[starting_point + offset:][1::2]
            new_edges = np.r_[original, adapted]
            new_binning = self.apply_adaptive_binning(
                components=components,
                bin_edges=new_edges,
                start_from=start_from,
                min_count=min_count
            )
        elif start_from == "right":
            starting_point = len(initial_hist) - np.argmax(np.flip(initial_hist) < min_count)
            offset = 0 if len(initial_hist[:starting_point]) % 2 == 0 else 1
            original = bin_edges[starting_point + offset:]
            adapted = bin_edges[:starting_point + offset][::2]
            new_edges = np.r_[adapted, original]
            new_binning = self.apply_adaptive_binning(
                components=components,
                bin_edges=new_edges,
                start_from=start_from,
                min_count=min_count
            )
        elif start_from == "max":
            max_bin = np.argmax(initial_hist)
            assert np.all(initial_hist[max_bin - 2:max_bin + 3] >= min_count)
            original_mid = bin_edges[max_bin - 1:max_bin + 2]
            adopted_left = self.apply_adaptive_binning(
                components=components,
                bin_edges=bin_edges[:max_bin - 1],
                start_from="right",
                min_count=min_count
            )[0]
            adopted_right = self.apply_adaptive_binning(
                components=components,
                bin_edges=bin_edges[max_bin + 2:],
                start_from="left",
                min_count=min_count
            )[0]
            new_edges = np.r_[adopted_left, original_mid, adopted_right]
            bin_mids = (new_edges[1:] + new_edges[:-1]) / 2
            bin_widths = new_edges[1:] - new_edges[:-1]
            new_binning = (new_edges, bin_mids, bin_widths)
        elif start_from == "auto":
            max_bin = np.argmax(initial_hist)
            if max_bin / len(initial_hist) < 0.15:
                method = "left"
            elif max_bin / len(initial_hist) > 0.85:
                method = "right"
            else:
                method = "max"
            return self.apply_adaptive_binning(
                components=components,
                bin_edges=bin_edges,
                start_from=method,
                min_count=min_count
            )
        else:
            raise ValueError(f"Value provided for parameter `start_from` is not valid.\n"
                             f"You provided '{start_from}'.\nShould be one of {valid_start_froms}.")

        assert new_binning[0][0] == bin_edges[0]
        assert new_binning[0][-1] == bin_edges[-1]

        return new_binning
