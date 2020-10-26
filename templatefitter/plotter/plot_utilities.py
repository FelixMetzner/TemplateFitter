"""
Provides some utility functions for matplotlib plots.
"""
import os
import copy
import logging
import tikzplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes._axes as axes
import matplotlib.colors as mpl_colors

from matplotlib import figure
from typing import Union, Tuple, AnyStr

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "AxesType",
    "FigureType",
    "export",
    "save_figure_as_tikz_tex_file",
    "color_fader"
]

AxesType = axes.Axes
FigureType = figure.Figure


def export(
        fig: plt.Figure,
        filename: Union[AnyStr, os.fspath],
        target_dir: Union[AnyStr, os.fspath] = "plots/",
        file_formats: Tuple[str, ...] = (".pdf", ".png"),
        save_as_tikz: bool = False,
        close_figure: bool = False
) -> None:
    """
    Convenience function for saving a matplotlib figure.

    :param fig: A matplotlib figure.
    :param filename: Filename of the plot without .pdf suffix.
    :param file_formats: Tuple of file formats specifying the format
                         figure will be saved as.
    :param target_dir: Directory where the plot will be saved in.
                       Default is './plots/'.
    :param save_as_tikz: Save the plot also as raw tikz tex document.
    :param close_figure: Whether to close the figure after saving it.
                         Default is False
    :return: None
    """
    os.makedirs(target_dir, exist_ok=True)

    for file_format in file_formats:
        fig.savefig(os.path.join(target_dir, f"{filename}{file_format}"), bbox_inches="tight")

    if save_as_tikz:
        save_figure_as_tikz_tex_file(fig=fig, target_path=os.path.join(target_dir, f"{filename}_tikz.tex"))

    if close_figure:
        plt.close(fig)
        fig.clf()


def save_figure_as_tikz_tex_file(fig: plt.Figure, target_path: Union[str, os.fspath]):
    try:
        tikzplotlib.clean_figure(fig=fig)
        tikzplotlib.save(figure=fig, filepath=target_path, strict=True)
    except Exception as e:
        logging.error(
            f"Exception ({e.__class__.__name__}) occurred in attempt to export plot in tikz raw text format!\n"
            f"The following tikz tex file was not produced.\n\t{target_path}\n"
            f"The following lines show additional information on the {e.__class__.__name__}",
            exc_info=e
        )


def color_fader(color_1: str, color_2: str, mix: float = 0.) -> str:
    c1 = np.array(mpl_colors.to_rgb(color_1))
    c2 = np.array(mpl_colors.to_rgb(color_2))
    return mpl_colors.to_hex((1 - mix) * c1 + mix * c2)
