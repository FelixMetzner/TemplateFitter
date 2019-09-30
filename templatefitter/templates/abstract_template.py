"""
Defines base class for templates used for fitting.
The basis for this is the more general BinnedDistribution
"""

import logging

from abc import ABC, abstractmethod

from templatefitter.binned_distributions.binning import BinsInputType, ScopeInputType
from templatefitter.binned_distributions.binned_distribution import BinnedDistribution

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["AbstractTemplate"]

# TODO: Check __init__ once all files have been refactored


class AbstractTemplate(ABC, BinnedDistribution):
    def __init__(self, name: str, dimensions: int, bins: BinsInputType, scope: ScopeInputType, params):
        BinnedDistribution.__init__(self, bins=bins, dimensions=dimensions, scope=scope, name=name)
        self._params = params

    @property
    def name(self):
        return self._name

    @property
    def num_bins(self):
        """int: Number of bins."""
        return self._num_bins

    @property
    def bins(self):
        """int or tuple of int: Number of bins."""
        return self._bins

    @property
    def range(self):
        """"""
        return self.range

    @abstractmethod
    def fractions(self):
        """ Abstract method which will return the bin fractions"""
        pass

    @abstractmethod
    def all_fractions(self):
        """ Abstract method which will return the bin fractions"""
        pass

    @abstractmethod
    def shape(self):
        """tuple of int: Template shape."""
        pass

    @abstractmethod
    def bin_mids(self):
        pass

    @abstractmethod
    def bin_edges(self):
        pass

    @abstractmethod
    def bin_widths(self):
        pass
