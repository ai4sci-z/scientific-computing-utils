"""
Visualization utilities for 2D flow fields.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_vector_field(x, y, u, v, title="Velocity field",
                      density=1.5, filename=None):
    """Streamline + speed color map."""
    speed = np.sqrt(u**2 + v**2)
    fig, ax = plt.subplots(figsize=(7, 6))
    strm = ax.streamplot(x, y, u.T, v.T, color=speed.T,
                         linewidth=1, cmap='viridis', density=density)
    plt.colorbar(strm.lines, ax=ax, label='|u|')
    ax.set_title(title); ax.set_xlabel('x'); ax.set_ylabel('y')
    plt.tight_layout()
    if filename: plt.savefig(filename, dpi=150)
    plt.show()


def plot_contour(x, y, field, levels=20, cmap='RdBu_r',
                 title="", label="", filename=None):
    """Filled contour plot."""
    X, Y = np.meshgrid(x, y, indexing='ij')
    fig, ax = plt.subplots(figsize=(6, 5))
    cf = ax.contourf(X, Y, field, levels=levels, cmap=cmap)
    plt.colorbar(cf, ax=ax, label=label)
    ax.contour(X, Y, field, levels=levels//4, colors='k', linewidths=0.4)
    ax.set_title(title); ax.set_xlabel('x'); ax.set_ylabel('y')
    plt.tight_layout()
    if filename: plt.savefig(filename, dpi=150)
    plt.show()


def plot_error_map(x, y, u_num, u_exact, title="Error map", filename=None):
    """Log10 absolute error heatmap."""
    err = np.abs(u_num - u_exact)
    err = np.where(err > 0, np.log10(err + 1e-16), -16)
    plot_contour(x, y, err, cmap='hot_r',
                 title=title, label='log10|error|', filename=filename)


def multi_snapshot(fields, times, cmap='RdBu_r', title="", filename=None):
    """Grid of snapshots for time evolution."""
    n = len(fields)
    fig, axes = plt.subplots(1, n, figsize=(3.5*n, 4))
    if n == 1: axes = [axes]
    vmin = min(f.min() for f in fields)
    vmax = max(f.max() for f in fields)
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax) if vmin < 0 < vmax else None
    for ax, field, t in zip(axes, fields, times):
        im = ax.imshow(field.T, origin='lower', cmap=cmap, norm=norm, aspect='auto')
        ax.set_title(f't = {t:.2f}')
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    plt.suptitle(title)
    plt.tight_layout()
    if filename: plt.savefig(filename, dpi=150)
    plt.show()
