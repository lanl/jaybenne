#!/usr/bin/env python3
# ========================================================================================
#  (C) (or copyright) 2024. Triad National Security, LLC. All rights reserved.
#
#  This program was produced under U.S. Government contract 89233218CNA000001 for Los
#  Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
#  for the U.S. Department of Energy/National Nuclear Security Administration. All rights
#  in the program are reserved by Triad National Security, LLC, and the U.S. Department
#  of Energy/National Nuclear Security Administration. The Government is granted for
#  itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
#  license in this material to reproduce, prepare derivative works, distribute copies to
#  the public, perform publicly and display publicly, and to permit others to do so.
# ========================================================================================

import os
import sys
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import jhdf


# Plot a 1D profile of a variable
def plot_1d(
    fig, ax, filenames, variable_name, draw_meshblocks, vmin, vmax, coords, scale
):

    for filename in filenames:

        dump = jhdf.jhdf(filename)

        # Indices outside of slice
        idx_k = 0
        idx_j = 0

        variable = dump.Get(variable_name)
        assert variable is not None, f"Variable {variable_name} does not exist!"
        if scale == "log":
            variable = np.log10(variable)

        for b in range(dump.NumBlocks):
            ax.plot(dump.xc[b, idx_k, idx_j, :], variable[b, idx_k, idx_j, :])


# Plot a 2D slice of a variable
def plot_2d(
    fig,
    ax,
    filename,
    variable_name,
    draw_particles,
    particle_reduction_factor,
    draw_meshblocks,
    vmin,
    vmax,
    coords,
    scale,
):
    dump = jhdf.jhdf(filename[0])

    assert len(filename) == 1, "Only one file can be plotted in 2D!"
    assert dump.NumDims >= 2, "1D data cannot be plotted in 2D!"

    # Indices outside of slice
    idx_k = 0

    variable = dump.Get(variable_name)
    assert variable is not None, f"Variable {variable_name} does not exist!"
    if scale == "log":
        variable = np.log10(variable)

    for b in range(dump.NumBlocks):
        ax.pcolormesh(
            dump.xc[b, idx_k, :, :],
            dump.yc[b, idx_k, :, :],
            variable[b, idx_k, :, :],
            vmin=vmin,
            vmax=vmax,
        )

    if draw_particles:
        photons = dump.GetSwarm("photons")
        id = photons.Get("id")
        # Construct reduced list
        x = []
        y = []
        for n in range(len(id)):
            if id[n] % particle_reduction_factor == 0:
                x.append(photons.x[n])
                y.append(photons.y[n])
        ax.plot(x, y, linestyle="", marker=".", color="k", markersize=1, alpha=0.5)

    if draw_meshblocks:
        color = "k"
        lw = 1.0
        alpha = 0.2
        for b in range(dump.NumBlocks):
            # Slice is always xy for now
            # TODO(BRR) choose a z position for slice, default to z=zmax?
            ax.plot(
                dump.xn[b, 0, 0, :],
                dump.yn[b, 0, 0, :],
                color=color,
                linewidth=lw,
                alpha=alpha,
            )
            ax.plot(
                dump.xn[b, 0, :, 0],
                dump.yn[b, 0, :, 0],
                color=color,
                linewidth=lw,
                alpha=alpha,
            )
            ax.plot(
                dump.xn[b, 0, -1, :],
                dump.yn[b, 0, -1, :],
                color=color,
                linewidth=lw,
                alpha=alpha,
            )
            ax.plot(
                dump.xn[b, 0, :, -1],
                dump.yn[b, 0, :, -1],
                color=color,
                linewidth=lw,
                alpha=alpha,
            )

    ax.set_aspect("equal")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot McBlock output")
    parser.add_argument(
        "filename", nargs="+", type=str, help="McBlock output file(s) to plot"
    )
    parser.add_argument(
        "variable", type=str, default="temperature", help="Variable name to plot"
    )
    parser.add_argument(
        "--dim",
        type=int,
        choices=[1, 2],
        default=1,
        help="Dimension (1 or 2) in which to plot",
    )
    parser.add_argument("--particles", action="store_true", help="Draw particles")
    parser.add_argument(
        "--particle_reduce",
        type=int,
        default=1,
        help="Factor by which to reduce drawn particles",
    )
    parser.add_argument(
        "--meshblocks", action="store_true", help="Draw meshblock boundaries"
    )
    parser.add_argument(
        "--vmin", type=float, default=-5, help="Minimum value of colorbar"
    )
    parser.add_argument(
        "--vmax", type=float, default=0, help="Maximum value of colorbar"
    )
    parser.add_argument(
        "--scale",
        type=str,
        default="log",
        choices=["linear", "log"],
        help="Scale in which to plot variable.",
    )
    parser.add_argument(
        "--coords",
        type=str,
        default="cartesian",
        help="Coordinates to plot. Choices are: [cartesian code]",
    )
    args = parser.parse_args()

    fig, ax = plt.subplots(1, 1)

    if args.dim == 1:
        plot_1d(
            fig,
            ax,
            args.filename,
            args.variable,
            args.meshblocks,
            args.vmin,
            args.vmax,
            args.coords,
            args.scale,
        )
    elif args.dim == 2:
        plot_2d(
            fig,
            ax,
            args.filename,
            args.variable,
            args.particles,
            args.particle_reduce,
            args.meshblocks,
            args.vmin,
            args.vmax,
            args.coords,
            args.scale,
        )

    plt.savefig(
        os.path.basename(args.filename[0])[:-5] + ".png", dpi=300, bbox_inches="tight"
    )
