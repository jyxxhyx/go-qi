import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon


import functools

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon


def draw_solution(grid, blocks1, blocks2, file_name):
    width = 10
    grid_nodes = [[[i, j], [i + 1, j], [i + 1, j + 1], [i, j + 1], [i, j]]
                  for i in range(grid[0]) for j in range(grid[1])]
    blocks1 = [[[i, j], [i + 1, j], [i + 1, j + 1], [i, j + 1], [i, j]]
                  for (i, j) in blocks1]
    blocks2 = [[[i, j], [i + 1, j], [i + 1, j + 1], [i, j + 1], [i, j]]
               for (i, j) in blocks2]
    blocks2 = sorted(blocks2, key=functools.cmp_to_key(compare))
    fig, ax = plt.subplots()

    _draw_blocks(grid_nodes, width, ax, 0.5, 'w')
    _draw_blocks(blocks1, width, ax, 2, 'darkblue')
    _draw_blocks(blocks2, width, ax, 2, 'skyblue')
    _draw_block_numbers(blocks2, width, fig, 2, 'black')

    ax.set_xlim([0, grid[0] * width])
    ax.set_ylim([0, grid[1] * width])
    ax.axis('equal')
    ax.set_axis_off()
    plt.margins(0, 0)
    plt.savefig('{}.jpg'.format(file_name), bbox_inches='tight', pad_inches=0)
    plt.savefig('{}.pdf'.format(file_name), bbox_inches='tight', pad_inches=0)


def _draw_blocks(blocks, width, ax, line_width, face_color):
    polygons = list()
    for block in blocks:
        polygon = [(x * width, y * width) for (x, y) in block]
        polygons.append(polygon)
    patches = [Polygon(polygon) for polygon in polygons]
    p = PatchCollection(patches,
                        facecolors=face_color,
                        edgecolors='k',
                        linewidths=line_width,
                        alpha=0.6)
    ax.add_collection(p)
    return


def _draw_block_numbers(blocks, grid_width, fig, line_width, face_color):
    for idx, rec in enumerate(blocks):
        x = (min(node[0]
                 for node in rec) + max(node[0]
                                        for node in rec)) / 2 * grid_width
        y = (min(node[1]
                 for node in rec) + max(node[1]
                                        for node in rec)) / 2 * grid_width
        size = grid_width * 2
        plt.text(x,
                 y,
                 str(idx),
                 fontsize=size,
                 figure=fig,
                 fontweight='extra bold',
                 fontfamily='monospace',
                 va='center',
                 ha='center',
                 c=face_color)
    return


def compare(block1, block2):
    max_col_1 = max(item[1] for item in block1)
    max_col_2 = max(item[1] for item in block2)
    if max_col_1 != max_col_2:
        return -max_col_1 + max_col_2
    else:
        return max(item[0] for item in block1) - max(item[0]
                                                     for item in block2)
