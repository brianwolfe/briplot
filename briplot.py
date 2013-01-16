#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np

def set_square(fig, width=1.8, **kwargs):
    # Set to square
    set_rectangular(width=width, aspect=1.0, **kwargs)

def set_rectangular(fig, width=1.8, aspect=1.6, n_columns=1,
                    padding=[0, 0, 0, 0], points=None):
    """
    Set all the axes inside to have width of width and an aspect ratio
    (width / height) of aspect
    """

    ax_list = fig._get_axes()

    if len(ax_list) == 0:
        fig.add_subplot(111)
        ax_list = fig.get_axes()

    # Figure out the layout
    n_rows = len(ax_list) / n_columns

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

    tot_width = per_ax_width * n_columns
    tot_height = per_ax_height * n_rows

    for i in range(n_rows):
        for j in range(n_columns):
            box = [
                (per_ax_width * j + padding[0] +
                    points[0][0] * per_ax_width) / tot_width,
                (per_ax_height * (n_rows - i - 1) + padding[1] +
                    points[0][1] * per_ax_height) / tot_height,
                width / tot_width ,
                height / tot_height
                ]
            print box
            ax_list[i * n_columns + j].set_position(box)

    # Set the figure size
    fig.set_size_inches(tot_width, tot_height)

def cumhist(*args, **kwargs):
    pass

