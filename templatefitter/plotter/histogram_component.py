"""
TODO
"""

import numpy as np

from typing import Optional

from templatefitter.binned_distributions.systematics import SystematicsInfo

__all__ = [
    "HistComponent"
]


class HistComponent:
    """
    Helper class for handling components of histograms.
    """

    def __init__(
            self,
            label: str,
            data: np.ndarray,
            weights: Optional[np.ndarray],
            systematics: Optional[SystematicsInfo] = None,
            hist_type: Optional[str] = None,
            color: Optional[str] = None,
            alpha: float = 1.0
    ):
        """
        HistComponent constructor.

        :param label: Component label for the histogram.
        :param data: Data to be plotted as histogram.
        :param weights: Weights for the events in data.
        :param systematics: Information about the systematics associated with the data.
        :param hist_type: Specifies the histogram type of the component in the histogram.
        :param color: Color of the histogram component.
        :param alpha: Alpha value of the histogram component.
        """
        self._label = label
        self._data = data
        self._weights = weights
        self._systematics = systematics
        self._hist_type = hist_type
        self._color = color
        self._min = np.amin(data) if len(data) > 0 else +float("inf")
        self._max = np.amax(data) if len(data) > 0 else -float("inf")
        self._alpha = alpha

    @property
    def label(self) -> str:
        return self._label

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def weights(self) -> Optional[np.ndarray]:
        return self._weights

    @property
    def systematics(self) -> Optional[SystematicsInfo]:
        return self._systematics

    @property
    def hist_type(self) -> Optional[str]:
        return self._hist_type

    @property
    def color(self) -> Optional[str]:
        return self._color

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def min_val(self) -> float:
        return self._min

    @property
    def max_val(self) -> float:
        return self._max
