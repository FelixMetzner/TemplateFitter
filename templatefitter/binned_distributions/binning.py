# Class providing general methods for binning in arbitrary dimensions.

import numpy as np
from typing import Union, Tuple, Optional


class Binning:
    def __init__(
            self,
            bins: Union[int, Tuple[int, ...], Tuple[float, ...], Tuple[Tuple[float, ...], ...]],
            dimensions: int,
            scope: Optional[Tuple[float, float], Tuple[Tuple[float, float], ...]] = None
    ):
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
            bins_input: Union[int, Tuple[int, ...], Tuple[float, ...], Tuple[Tuple[float, ...], ...]],
            scope_input: Optional[Tuple[float, float], Tuple[Tuple[float, float], ...]] = None
    ):
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
                self._num_bins = (bins_input,)
                self._bin_edges = (tuple(np.linspace(*scope_input, bins_input + 1)),)
            elif isinstance(bins_input, tuple) and all(isinstance(bin_num, float) for bin_num in bins_input):
                self._num_bins = (len(bins_input) - 1,)
                self._bin_edges = (bins_input,)
                if scope_input is not None:
                    assert scope_input[0] == bins_input[0], scope_error_txt
                    assert scope_input[1] == bins_input[-1], scope_error_txt
            else:
                raise ValueError(error_txt)
        else:
            if not isinstance(bins_input, tuple) or self.dimensions != len(bins_input):
                raise ValueError(error_txt)
            if all(isinstance(bin_num, int) for bin_num in bins_input):
                if scope_input is None or len(bins_input) != len(scope_input):
                    raise ValueError(num_error_txt)
                self._num_bins = bins_input
                self._bin_edges = (tuple(np.linspace(*scp, num + 1)) for num, scp in zip(bins_input, scope_input))
            elif all(isinstance(bin_num, tuple) for bin_num in bins_input):
                assert all(isinstance(edge, float) for edges in bins_input for edge in edges), bins_input
                if scope_input is not None:
                    assert all(scp[0] == edges[0] for scp, edges in zip(scope_input, bins_input)), scope_error_txt
                    assert all(scp[1] == edges[-1] for scp, edges in zip(scope_input, bins_input)), scope_error_txt
                self._num_bins = (len(edges) - 1 for edges in bins_input)
                self._bin_edges = bins_input
            else:
                raise ValueError(error_txt)

        self._bin_mids = tuple(map(self._get_bin_mids, self.bin_edges))
        self._bin_widths = tuple(map(self._get_bin_widths, self.bin_edges))
        self._range = tuple(map(self._get_range, self.bin_edges))

    def _check_binning(self):
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
    def _get_bin_mids(bin_edges):
        return tuple((np.array(bin_edges)[1:] + np.array(bin_edges)[:-1]) / 2.)

    @staticmethod
    def _get_bin_widths(bin_edges):
        return tuple(np.array(bin_edges)[1:] - np.array(bin_edges)[:-1])

    @staticmethod
    def _get_range(bin_edges):
        return bin_edges[0], bin_edges[-1]

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def num_bins(self) -> Tuple[int, ...]:
        return self._num_bins

    def num_bins_total(self) -> int:
        return sum(self._num_bins)

    @property
    def bin_edges(self) -> Tuple[Tuple[float, ...]]:
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
    def range(self) -> Tuple[Tuple[float, ...]]:
        return self._range
