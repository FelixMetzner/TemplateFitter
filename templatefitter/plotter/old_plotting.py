# Basic plotting used by Will and Max

import numpy as np
from typing import NamedTuple, Callable

from templatefitter.fit_model.parameter_handler import ParameterHandler
from templatefitter.histograms.abstract_hist import AbstractHist


# TODO: Maybe add also:
#           - latex_tables.py
#           - mva_plots.py


class PlottingInfo(NamedTuple):  # TODO: This is not used atm... is it necessary? Maybe for template to hist conversion?
    templates: dict
    params: ParameterHandler
    yield_indices: list
    dimension: int
    projection_fct: Callable
    data: AbstractHist
    has_data: bool


def plot_stacked_on(plot_info, ax, plot_all=False, **kwargs):
    bin_mids = [template.bin_mids() for template in plot_info.templates.values()]
    bin_edges = next(iter(plot_info.templates.values())).bin_edges()
    bin_width = next(iter(plot_info.templates.values())).bin_widths()
    num_bins = next(iter(plot_info.templates.values())).num_bins
    shape = next(iter(plot_info.templates.values())).shape()

    colors = [template.color for template in plot_info.templates.values()]
    yields = plot_info.params.get_parameters([plot_info.yield_indices])
    bin_counts = [temp_yield * temp.fractions() for temp_yield, temp in zip(yields, plot_info.templates.values())]
    labels = [template.name for template in plot_info.templates.values()]

    if plot_all:
        colors = []
        for template in plot_info.templates.values():
            colors += template.colors()
        labels = []
        for template in plot_info.templates.values():
            labels += template.labels()
        bin_counts = [
            temp_yield * template.all_fractions() for temp_yield, template in zip(yields, plot_info.templates.values())
        ]
        bin_counts = np.concatenate(bin_counts)
        N = len(bin_counts)
        bin_counts = np.split(bin_counts, N / num_bins)
        bin_mids = [bin_mids[0]] * int(N / num_bins)

    if plot_info.dimension > 1:
        bin_counts = [plot_info.projection(kwargs["projection"], bc.reshape(shape)) for bc in bin_counts]
        axis = kwargs["projection"]
        ax_to_index = {
            "x": 0,
            "y": 1,
        }
        bin_mids = [mids[ax_to_index[axis]] for mids in bin_mids]
        bin_edges = bin_edges[ax_to_index[axis]]
        bin_width = bin_width[ax_to_index[axis]]

    ax.hist(
        bin_mids,
        weights=bin_counts,
        bins=bin_edges,
        edgecolor="black",
        histtype="stepfilled",
        lw=0.5,
        color=colors,
        label=labels,
        stacked=True,
    )

    uncertainties_sq = [
        (temp_yield * template.fractions() * template.errors()).reshape(template.shape()) ** 2
        for temp_yield, template in zip(yields, plot_info.templates.values())
    ]
    if plot_info.dimension > 1:
        uncertainties_sq = [plot_info.projection(kwargs["projection"], unc_sq) for unc_sq in uncertainties_sq]

    total_uncertainty = np.sqrt(np.sum(np.array(uncertainties_sq), axis=0))
    total_bin_count = np.sum(np.array(bin_counts), axis=0)

    ax.bar(
        x=bin_mids[0],
        height=2 * total_uncertainty,
        width=bin_width,
        bottom=total_bin_count - total_uncertainty,
        color="black",
        hatch="///////",
        fill=False,
        lw=0,
        label="MC Uncertainty",
    )

    if plot_info.data is None:
        return ax

    data_bin_mids = plot_info.data.bin_mids
    data_bin_counts = plot_info.data.bin_counts
    data_bin_errors_sq = plot_info.data.bin_errors_sq

    if plot_info.has_data:

        if plot_info.dimension > 1:
            data_bin_counts = plot_info.projection(kwargs["projection"], data_bin_counts)
            data_bin_errors_sq = plot_info.projection(kwargs["projection"], data_bin_errors_sq)

            axis = kwargs["projection"]
            ax_to_index = {
                "x": 0,
                "y": 1,
            }
            data_bin_mids = data_bin_mids[ax_to_index[axis]]

        ax.errorbar(
            x=data_bin_mids,
            y=data_bin_counts,
            yerr=np.sqrt(data_bin_errors_sq),
            ls="",
            marker=".",
            color="black",
            label="Data",
        )

    return ax
