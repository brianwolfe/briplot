#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import string
from copy import copy
from math import ceil
axislabels = list(string.ascii_lowercase) + [''.join((a, b)) for a in string.ascii_lowercase for b in string.ascii_lowercase]

def set_square(fig, width=1.9, **kwargs):
    # Set to square
    set_rectangular(width=width, aspect=1.0, **kwargs)

def set_rectangular(fig, width=1.75, aspect=1.6, n_columns=1,
                    padding=[0, 0, 0, 0], points=None, y_ax_pad=0.35,
                    x_ax_pad=0.25):
    """
    Set all the axes inside to have width of width and an aspect ratio
    (width / height) of aspect
    """

    ax_list = fig._get_axes()

    if len(ax_list) == 0:
        fig.add_subplot(111)
        ax_list = fig.get_axes()

    # Figure out the layout
    n_rows = int(ceil(float(len(ax_list)) / n_columns))

    # Figure out the size we should be
    if points is None:
        points = [[plt.rcParams['figure.subplot.left'],
                    plt.rcParams['figure.subplot.bottom']],
                  [plt.rcParams['figure.subplot.right'],
                    plt.rcParams['figure.subplot.top']]]
    norm_width = points[1][0] - points[0][0]
    norm_height = points[1][1] - points[0][1]
    height = width / aspect

    per_ax_width = width / norm_width
    per_ax_height = height / norm_height
    padding = copy(padding)

    if points[0][0] > 1 - points[1][0]:
        padding[2] += (points[0][0] - (1 - points[1][0])) * per_ax_width
    else:
        padding[0] += -(points[0][0] - (1 - points[1][0])) * per_ax_width
    if points[0][1] > 1 - points[1][1]:
        padding[3] += (points[0][1] - (1 - points[1][1])) * per_ax_height
    else:
        padding[1] += -(points[0][1]) - (1 - points[1][1]) * per_ax_height


    tot_width = per_ax_width * (n_columns) + padding[0] + padding[2]
    tot_height = per_ax_height * (n_rows) + padding[1] + padding[3]

    x_ax_pad_n = x_ax_pad / per_ax_height
    y_ax_pad_n = y_ax_pad / per_ax_width

    for i in range(n_rows):
        for j in range(n_columns):
            cur_ax = i*n_columns + j
            if cur_ax < len(ax_list):
                box = [
                    (per_ax_width * j + padding[0] +
                        points[0][0] * per_ax_width) / tot_width,
                    (per_ax_height * (n_rows - i - 1) + padding[1] +
                        points[0][1] * per_ax_height) / tot_height,
                    width / tot_width ,
                    height / tot_height
                    ]
                ax_list[cur_ax].set_position(box)
                ax_list[cur_ax].xaxis.set_label_coords(0.5, -x_ax_pad_n)
                ax_list[cur_ax].yaxis.set_label_coords(-y_ax_pad_n, 0.5)
                ax_list[cur_ax].text(-y_ax_pad_n*1.15, 1.05, axislabels[cur_ax], transform=ax_list[cur_ax].transAxes,
                                    fontweight='bold', horizontalalignment='right',
                                    fontsize='large')
                grids = [tick.gridline for tick in ax_list[cur_ax].xaxis.get_major_ticks()]
                grids += [tick.gridline for tick in ax_list[cur_ax].xaxis.get_minor_ticks()]
                grids += [tick.gridline for tick in ax_list[cur_ax].yaxis.get_major_ticks()]
                grids += [tick.gridline for tick in ax_list[cur_ax].yaxis.get_minor_ticks()]
                for g in grids:
                    g.set_dashes((0.25, 2))

    # Set the figure size
    fig.set_size_inches(tot_width, tot_height)

def cumhist(ax, x, minx=None, maxx=None, n_bins=1000, logscale=True, normalized=True,
           **kwargs):

    if minx is None:
        minx = min(x)

    if maxx is None:
        maxx = max(x)

    nice = True
    if nice and not logscale :
        mlog = np.floor(np.log10(abs(maxx)))
        mlog -= 1
        precision = 10 ** mlog

        maxx = np.ceil(maxx / precision) * precision
        minx = np.floor(minx / precision) * precision

    if logscale and not minx > 0:
        raise ValueError("x must be > 0 for log scaled histogram")

    bins = []
    if logscale:
        lminx = np.floor(np.log10(minx))
        lmaxx = np.ceil(np.log10(maxx))
        bins = np.logspace(lminx, lmaxx, n_bins)
    else:
        r = maxx - minx
        bins = np.linspace(minx - r * 0.02, maxx + 0.02, n_bins)
    hist, bins = np.histogram(x, bins)
    n_samples = len(x)

    if normalized:
        hist = hist / float(n_samples)

    cumulative = np.cumsum(hist)

    h = ax.plot(bins[:-1], cumulative, **kwargs)
    ax.set_xlim(minx, maxx)
    if logscale:
        ax.set_xscale('log')
    else:
        ax.set_xscale('linear')

    if normalized:
        ax.set_ylim(0, 1)

    return h

