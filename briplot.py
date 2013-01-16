#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np

def set_square(fig, width=1.8, **kwargs):
    # Set to square
    set_rectangular(width=width, aspect=1.0, **kwargs)

def set_rectangular(fig, width=1.8, aspect=1.6, n_columns=1,
                    extrawidth=0.0, extraheight=0.0):
    """
    Set all the axes inside to have width of width and an aspect ratio
    (width / height) of aspect
    """

    ax_list = fig._get_axes()

    if len(ax_list) == 0:
        fig.add_subplot(111)
        ax_list = fig.get_axes()

    # Figure out the layout

    # Figure out the size we should be
    ax = ax_list[0]
    bbox = ax.get_position()
    points = bbox.get_points()
    norm_width = points[1][0] - points[0][0]
    norm_height = points[1][1] - points[0][1]

    tot_width = width / norm_width
    tot_height = width / aspect / norm_height

    # Set the figure size
    fig.set_size_inches(tot_width, tot_height)

def cumhist(*args, **kwargs):
    pass

x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
set_rectangular(plt.gcf())
plt.savefig('test.pdf')

