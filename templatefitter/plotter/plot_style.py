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

    kit_green: str = "#009682"
    kit_blue: str = "#4664aa"
    kit_maygreen: str = "#8cb63c"
    kit_yellow: str = "#fce500"
    kit_orange: str = "#df9b1b"
    kit_brown: str = "#a7822e"
    kit_red: str = "#a22223"
    kit_purple: str = "#a3107c"
    kit_cyan: str = "#23a1e0"
    kit_black: str = "#000000"
    white: str = "#ffffff"
    light_grey: str = "#bdbdbd"
    grey: str = "#797979"
    dark_grey: str = "#4e4e4e"

    default_colors: List[str] = [
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
    ]


kit_color_cycler = cycler("color", KITColors.default_colors)


class TangoColors:
    butter1: str = "#fce94f"
    butter2: str = "#edd400"
    butter3: str = "#c4a000"
    orange1: str = "#fcaf3e"
    orange2: str = "#f57900"
    orange3: str = "#ce5c00"
    chocolate1: str = "#e9b96e"
    chocolate2: str = "#c17d11"
    chocolate3: str = "#8f5902"
    chameleon1: str = "#8ae234"
    chameleon2: str = "#73d216"
    chameleon3: str = "#4e9a06"
    skyblue1: str = "#729fcf"
    skyblue2: str = "#3465a4"
    skyblue3: str = "#204a87"
    plum1: str = "#ad7fa8"
    plum2: str = "#75507b"
    plum3: str = "#5c3566"
    scarletred1: str = "#ef2929"
    scarletred2: str = "#cc0000"
    scarletred3: str = "#a40000"
    aluminium1: str = "#eeeeec"
    aluminium2: str = "#d3d7cf"
    aluminium3: str = "#babdb6"
    aluminium4: str = "#888a85"
    aluminium5: str = "#555753"
    aluminium6: str = "#2e3436"

    default_colors: List[str] = [
        skyblue2,
        orange2,
        chameleon3,
        scarletred2,
        plum2,
        chocolate3,
        butter2,
        aluminium4,
        skyblue1,
        chameleon1,
    ]


tango_color_cycler = cycler("color", TangoColors.default_colors)


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
