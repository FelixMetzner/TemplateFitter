import logging

from abc import ABC, abstractmethod

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["AbstractTemplate"]


class AbstractTemplate(ABC):
    """
    Defines the template interface.
    """

    def __init__(self, name, params):
        self._name = name
        self._params = params
        self._bins = None
        self._num_bins = None
        self._range = None

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
    def allfractions(self):
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
