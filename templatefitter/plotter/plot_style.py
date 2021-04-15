"""
Provides color scheme and resets matplotlib default parameter setup
"""

import matplotlib.pyplot as plt

from typing import List
from cycler import cycler
from collections import OrderedDict

__all__ = [
    "xlabel_pos",
    "ylabel_pos",
    "KITColors",
    "kit_color_cycler",
    "set_matplotlibrc_params",
]

# The following dictionaries define variables which can be given
# as arguments to matplotlib"s set_x/y_label functions in order to
# align the axis-label text to the end of the axes.
xlabel_pos = OrderedDict([("x", 1), ("ha", "right")])
ylabel_pos = OrderedDict([("y", 1), ("ha", "right")])


class KITColors:
    """
    KIT color scheme plus additional grey shades
    """

    kit_green = "#009682"  # type: str
    kit_blue = "#4664aa"  # type: str
    kit_maygreen = "#8cb63c"  # type: str
    kit_yellow = "#fce500"  # type: str
    kit_orange = "#df9b1b"  # type: str
    kit_brown = "#a7822e"  # type: str
    kit_red = "#a22223"  # type: str
    kit_purple = "#a3107c"  # type: str
    kit_cyan = "#23a1e0"  # type: str
    kit_black = "#000000"  # type: str
    white = "#ffffff"  # type: str
    light_grey = "#bdbdbd"  # type: str
    grey = "#797979"  # type: str
    dark_grey = "#4e4e4e"  # type: str

    default_colors = [
        kit_blue,
        kit_orange,
        kit_green,
        kit_red,
        kit_purple,
        kit_brown,
        kit_yellow,
        dark_grey,
        kit_cyan,
        kit_maygreen,
    ]  # type: List[str]


kit_color_cycler = cycler("color", KITColors.default_colors)


def set_matplotlibrc_params() -> None:
    """
    Sets default parameters in the matplotlibrc.
    :return: None
    """
    xtick = {
        "top": True,
        "minor.visible": True,
        "direction": "in",
        "labelsize": 10,
    }

    ytick = {
        "right": True,
        "minor.visible": True,
        "direction": "in",
        "labelsize": 10,
    }

    axes = {
        "labelsize": 12,
        "prop_cycle": kit_color_cycler,
        "formatter.limits": (-4, 4),
        "formatter.use_mathtext": True,
        "titlesize": "large",
        "labelpad": 4.0,
    }
    lines = {
        "lw": 1.5,
    }
    legend = {
        "frameon": False,
    }

    plt.rc("lines", **lines)
    plt.rc("axes", **axes)
    plt.rc("xtick", **xtick)
    plt.rc("ytick", **ytick)
    plt.rc("legend", **legend)

    plt.rcParams.update(
        {
            "figure.autolayout": True,
        }
    )
