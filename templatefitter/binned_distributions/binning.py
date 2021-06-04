"""
Class providing general methods for binning in arbitrary dimensions.
"""

import logging
import numpy as np
from typing import Union, Optional, Tuple, List

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "Binning",
    "BinEdgesType",
    "BinsInputType",
    "ScopeInputType",
    "LogScaleInputType",
]

BinsInputType = Union[int, Tuple[int, ...], Tuple[float, ...], Tuple[Tuple[float, ...], ...]]
ScopeInputType = Union[None, Tuple[float, float], Tuple[Tuple[float, float], ...]]
BinEdgesType = Tuple[Tuple[float, ...], ...]
LogScaleInputType = Union[bool, List[bool], Tuple[bool, ...]]


class Binning:
    def __init__(
        self,
        bins: BinsInputType,
        dimensions: int,
        scope: ScopeInputType = None,
        log_scale: LogScaleInputType = False,
    ) -> None:
        assert isinstance(dimensions, int) and dimensions > 0, (
            f"Dimensions must be integer greater than 0, " f"you provided {dimensions} of type {type(dimensions)}!"
        )

        self._dimensions = dimensions  # type: int

        self._num_bins = None  # type: Optional[Tuple[int, ...]]
        self._bin_edges = None  # type: Optional[BinEdgesType]
        self._bin_mids = None  # type: Optional[BinEdgesType]
        self._bin_widths = None  # type: Optional[BinEdgesType]
        self._range = None  # type: Optional[Tuple[Tuple[float, float], ...]]
        self._log_scale_mask = None  # type: Optional[Tuple[bool, ...]]

        self._init_log_scale(log_scale=log_scale)
        self._init_binning(bins_input=bins, scope_input=scope)
        self._check_binning()

    def _init_binning(
        self,
        bins_input: BinsInputType,
        scope_input: ScopeInputType,
    ) -> None:
        error_txt = (
            f"Ill defined binning for {self.dimensions} dimensions:\n"
            f"bins = {bins_input}\ntype(bins) = {type(bins_input)}"
        )
        num_error_txt = (
            f"If binning is defined via number of bins, scope is required, too!\n"
            f"bins = {bins_input}\n"
            f"type(bins) = {type(bins_input)}"
        )
        scope_error_txt = (
            f"The provided bin edges and range do not match!\n" f"Bin edges: {bins_input}\nRange: {scope_input}"
        )

        if self.dimensions == 1:
            if isinstance(bins_input, int):
                if scope_input is None:
                    raise ValueError(num_error_txt)
                assert isinstance(scope_input, tuple) and len(scope_input) == 2, (type(scope_input), scope_input)
                assert all(isinstance(scp, (float, int)) for scp in scope_input), scope_input
                _scope = scope_input  # type: Tuple[float, float]  # type: ignore
                self._num_bins = (bins_input,)
                self._bin_edges = (self._get_bin_edges(scope=_scope, bins=bins_input, log=self.log_scale_mask[0]),)
            elif isinstance(bins_input, tuple):
                if all(isinstance(bin_num, float) for bin_num in bins_input):
                    _bins_input = bins_input  # type: Tuple[float, ...]  # type: ignore
                    self._num_bins = (len(_bins_input) - 1,)
                    self._bin_edges = (_bins_input,)
                elif len(bins_input) == 1 and isinstance(bins_input[0], int):
                    if scope_input is None:
                        raise ValueError(num_error_txt)
                    assert isinstance(scope_input, tuple), (type(scope_input), scope_input)
                    if len(scope_input) == 1 and isinstance(scope_input[0], tuple):
                        assert len(scope_input[0]) == 2, scope_input
                        assert all(isinstance(scp, (float, int)) for scp in scope_input[0]), scope_input
                        scope_1d = scope_input[0]  # type: Tuple[float, float]
                    elif len(scope_input) == 2 and all(isinstance(scp, (float, int)) for scp in scope_input):
                        _scope_1d = scope_input  # type: Tuple[float, float]  # type: ignore
                        scope_1d = _scope_1d
                    else:
                        raise ValueError(scope_error_txt)
                    self._num_bins = (bins_input[0],)
                    edges = self._get_bin_edges(scope=scope_1d, bins=bins_input[0], log=self.log_scale_mask[0])
                    self._bin_edges = (edges,)
                elif (
                    all(isinstance(bi, tuple) for bi in bins_input)
                    and len(bins_input) == 1
                    and all(isinstance(be, float) for be in bins_input[0])  # type: ignore
                ):
                    _bins_in_first = bins_input[0]  # type: Tuple[float, ...]  # type: ignore
                    _bins_in_all = bins_input  # type: Tuple[Tuple[float, ...]]  # type: ignore
                    self._num_bins = (len(_bins_in_first) - 1,)
                    self._bin_edges = _bins_in_all
                else:
                    raise ValueError(error_txt)

                if scope_input is not None:
                    if isinstance(scope_input, tuple):
                        if all(isinstance(scp, float) for scp in scope_input) and len(scope_input) == 2:
                            assert scope_input[0] == self._bin_edges[0][0], (scope_input, self._bin_edges)
                            assert scope_input[1] == self._bin_edges[0][-1], (scope_input, self._bin_edges)
                        elif (
                            all(isinstance(scp, tuple) for scp in scope_input)
                            and len(scope_input) == 1
                            and all(isinstance(scp, float) for scp in scope_input[0])  # type: ignore
                        ):
                            _scope_in = scope_input[0]  # type: Tuple[float, float]  # type: ignore
                            assert _scope_in[0] == self._bin_edges[0][0], (scope_input, self._bin_edges)
                            assert _scope_in[1] == self._bin_edges[0][-1], (scope_input, self._bin_edges)
                        else:
                            raise ValueError(scope_error_txt)
                    else:
                        raise ValueError(scope_error_txt)
            else:
                raise ValueError(error_txt)
        else:
            if not isinstance(bins_input, tuple) or self.dimensions != len(bins_input):
                raise ValueError(error_txt)
            if all(isinstance(bin_num, int) for bin_num in bins_input):
                if scope_input is None or len(bins_input) != len(scope_input) or not isinstance(scope_input, tuple):
                    raise ValueError(num_error_txt)
                assert all(isinstance(scp, tuple) and len(scp) == 2 for scp in scope_input), scope_input
                _nd_bins_in = bins_input  # type: Tuple[int, ...]  # type: ignore
                _nd_scope_input = scope_input  # type: Tuple[Tuple[float, float], ...]  # type: ignore
                self._num_bins = _nd_bins_in
                self._bin_edges = tuple(
                    [
                        self._get_bin_edges(scope=scp, bins=num, log=log)
                        for num, scp, log in zip(_nd_bins_in, _nd_scope_input, self.log_scale_mask)
                    ]
                )
            elif all(isinstance(bin_num, tuple) for bin_num in bins_input):
                _nd_bins_input = bins_input  # type: Tuple[Tuple[float, ...], ...]  # type: ignore
                assert all(isinstance(edge, float) for edges in _nd_bins_input for edge in edges), _nd_bins_input
                if scope_input is not None:
                    _nd_scope_in = scope_input  # type: Tuple[Tuple[float, float], ...]  # type: ignore
                    # Just checking if provided scope fits the provided edges, not using it in this case...
                    assert isinstance(_nd_scope_in, tuple), (type(_nd_scope_in), _nd_scope_in)
                    assert all(isinstance(scp, tuple) and len(scp) == 2 for scp in _nd_scope_in), _nd_scope_in
                    assert all(scp[0] == edges[0] for scp, edges in zip(_nd_scope_in, _nd_bins_input)), scope_error_txt
                    assert all(scp[1] == edges[-1] for scp, edges in zip(_nd_scope_in, _nd_bins_input)), scope_error_txt
                self._num_bins = tuple([len(edges) - 1 for edges in _nd_bins_input])
                self._bin_edges = _nd_bins_input
            else:
                raise ValueError(error_txt)

        self._bin_mids = tuple(map(self._get_bin_mids, self.bin_edges))
        self._bin_widths = tuple(map(self._get_bin_widths, self.bin_edges))
        self._range = tuple(map(self._get_range, self.bin_edges))

    def _init_log_scale(
        self,
        log_scale: LogScaleInputType,
    ) -> None:
        assert self._log_scale_mask is None
        base_error_text = (
            "The argument 'log_scale', which can be used to define a logarithmic binning "
            "for some or all dimensions of the binning, must be either a boolean or a "
            "list or tuple of booleans"
        )

        if isinstance(log_scale, bool):
            self._log_scale_mask = tuple([log_scale for _ in range(self.dimensions)])
        elif isinstance(log_scale, list) or isinstance(log_scale, tuple):
            if not all(isinstance(ls, bool) for ls in log_scale):
                raise ValueError(
                    f"{base_error_text}, but you provided a {type(log_scale)} containing "
                    f"objects of the following types:\n{[type(ls) for ls in log_scale]}"
                )
            if len(log_scale) != self.dimensions:
                raise ValueError(
                    f"{base_error_text}.\nYou provided a {type(log_scale)} of booleans, but"
                    f"it is of length {len(log_scale)} for a binning meant for {self.dimensions} "
                    f"dimensions...\nThe length of the list/tuple must be the same as the number"
                    f"of dimensions!"
                )
            self._log_scale_mask = tuple(log_scale)
        else:
            raise ValueError(f"{base_error_text}.\n However, you provided an object of type {type(log_scale)}...")

    @staticmethod
    def _get_bin_edges(
        scope: Tuple[float, float],
        bins: int,
        log: bool,
    ) -> Tuple[float, ...]:
        if log:
            if not scope[0] > 0.0:
                raise RuntimeError(
                    f"Logarithmic binning cannot be used for a distribution with values <= 0!\n"
                    f"\tscope: {scope}\n\tnumber of bins: {bins}\n\tlog_scale bool: {log}"
                )
            return tuple(np.logspace(np.log10(scope[0]), np.log10(scope[1]), bins + 1))
        else:
            return tuple(np.linspace(*scope, bins + 1))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Binning):
            raise TypeError(f"Object on the right side is not of the type 'Binning', but of type {type(other).__name__}")

        if not self.dimensions == other.dimensions:
            return False
        if not self.num_bins == other.num_bins:
            return False
        if not self.bin_edges == other.bin_edges:
            return False
        if not self.range == other.range:
            return False
        if not self.log_scale_mask == other.log_scale_mask:
            return False

        return True

    def _check_binning(self) -> None:
        assert self._num_bins is not None, "Number of bins is not defined after initialization!"
        assert self._bin_edges is not None, "Bin edges are not defined after initialization!"

        assert isinstance(self._bin_edges, tuple)
        assert len(self._bin_edges) == self.dimensions, (len(self._bin_edges), self.dimensions)

        assert isinstance(self._num_bins, tuple), self._num_bins
        assert all(isinstance(n, int) for n in self._num_bins), self._num_bins
        assert len(self._num_bins) == self.dimensions, (len(self._num_bins), self.dimensions)
        assert all(len(self._bin_edges[i]) == bins + 1 for i, bins in enumerate(self._num_bins)), (
            self._num_bins,
            self._bin_edges,
        )

        assert len(self.bin_mids) == self.dimensions, (len(self.bin_mids), self.dimensions)
        assert all(len(m) == b for m, b in zip(self.bin_mids, self._num_bins))
        assert len(self.bin_widths) == self.dimensions, (len(self.bin_widths), self.dimensions)
        assert all(len(w) == b for w, b in zip(self.bin_widths, self._num_bins))
        assert len(self.range) == self.dimensions, (len(self.range), self.dimensions)
        assert all(len(r) == 2 for r in self.range)

    @staticmethod
    def _get_bin_mids(bin_edges: Tuple[float, ...]) -> Tuple[float, ...]:
        return tuple((np.array(bin_edges)[1:] + np.array(bin_edges)[:-1]) / 2.0)

    @staticmethod
    def _get_bin_widths(bin_edges: Tuple[float, ...]) -> Tuple[float, ...]:
        return tuple(np.array(bin_edges)[1:] - np.array(bin_edges)[:-1])

    @staticmethod
    def _get_range(bin_edges: Tuple[float, ...]) -> Tuple[float, float]:
        return bin_edges[0], bin_edges[-1]

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def num_bins(self) -> Tuple[int, ...]:
        assert self._num_bins is not None
        return self._num_bins

    @property
    def num_bins_total(self) -> int:
        return int(np.prod(self.num_bins))

    @property
    def bin_edges(self) -> BinEdgesType:
        assert self._bin_edges is not None
        return self._bin_edges

    @property
    def bin_edges_flattened(self) -> np.ndarray:
        return np.array([edge for edges in self.bin_edges for edge in edges])

    @property
    def bin_mids(self) -> Tuple[Tuple[float, ...], ...]:
        assert self._bin_mids is not None
        return self._bin_mids

    @property
    def bin_widths(self) -> Tuple[Tuple[float, ...], ...]:
        assert self._bin_widths is not None
        return self._bin_widths

    @property
    def range(self) -> Tuple[Tuple[float, float], ...]:
        assert self._range is not None
        return self._range

    @property
    def log_scale_mask(self) -> Tuple[bool, ...]:
        assert self._log_scale_mask is not None
        return self._log_scale_mask

    @property
    def as_string_list(self) -> List[str]:
        string_list = [
            f"dimensions = {self.dimensions}",
            f"number of bins per dimension = {self.num_bins}",
            f"total number of bins = {self.num_bins_total}",
            f"range = {self.range}",
            f"loc_scale_mask = {self.log_scale_mask}",
            f"bin_edges = {self.bin_edges}",
        ]
        return string_list

    def get_binning_for_one_dimension(self, dimension: int) -> "Binning":
        assert dimension < self.dimensions, (dimension, self.dimensions)

        if self.dimensions == 1 and dimension == 0:
            return self

        return Binning(
            bins=self.bin_edges[dimension],
            dimensions=1,
            scope=self.range[dimension],
            log_scale=self.log_scale_mask[dimension],
        )

    def get_binning_for_x_dimensions(self, dimensions: Tuple[int, ...]) -> "Binning":
        assert isinstance(dimensions, tuple), (dimensions, type(dimensions).__name__)
        assert all(isinstance(dim, int) for dim in dimensions), (dimensions, [type(d).__name__ for d in dimensions])
        assert all(0 <= dim < self.dimensions for dim in dimensions), (dimensions, self.dimensions)
        assert tuple(set(dimensions)) == dimensions, dimensions

        if len(dimensions) == self.dimensions:
            return self

        return Binning(
            bins=tuple([self.bin_edges[dim] for dim in dimensions]),
            dimensions=len(dimensions),
            scope=tuple([self.range[dim] for dim in dimensions]),
            log_scale=tuple([self.log_scale_mask[dim] for dim in dimensions]),
        )

    @staticmethod
    def _calc_bin_scaling(bin_widths: Tuple[float, ...]) -> np.ndarray:
        min_bin_width = min(bin_widths)  # type: float
        bin_widths_array = np.array(bin_widths)  # type: np.ndarray
        if all(bw == min_bin_width for bw in bin_widths):
            return np.ones_like(bin_widths_array)
        else:
            return 1.0 / np.around(bin_widths_array / min_bin_width, decimals=0)

    def get_bin_scaling_per_dim_tuple(self) -> Tuple[np.ndarray, ...]:
        return tuple([self._calc_bin_scaling(bin_widths=bws) for bws in self.bin_widths])

    def get_bin_scaling_for_dim(self, dimension: int) -> np.ndarray:
        if dimension >= self.dimensions or dimension < 0:
            raise ValueError(
                f"Argument 'dimension' must be an integer in [0, {self.dimensions - 1}], but {dimension} was provided!"
            )
        return self.get_bin_scaling_per_dim_tuple()[dimension]

    def get_bin_scaling(self) -> np.ndarray:
        if not self.dimensions == 1:
            raise RuntimeError(
                f"This function is only valid for 1 dimensional Binnings instances, "
                f"but this instance has {self.dimensions} dimensions!"
            )
        return self.get_bin_scaling_for_dim(dimension=0)
