from typing import Tuple, List

import pulp
from pulp import PULP_CBC_CMD, GUROBI_CMD

from model.abstract_model import AbstractModel


class GoQi(AbstractModel):
    def __init__(self, grid: Tuple[int, int], num_go: int):
        self.grid = grid
        self.num_go = num_go
        self.m = pulp.LpProblem('Qi', pulp.LpMinimize)
        return

    def _set_iterables(self):
        self.cells = [(i, j) for i in range(self.grid[0])
                      for j in range(self.grid[1])]
        self.neighbors = {(i, j): _generate_neighbors(i, j, self.grid)
                          for (i, j) in self.cells}
        self.boundary = [
            (i, j) for (i, j) in self.cells
            if (i == 0) or (i == self.grid[0] -
                            1) or (j == 0) or (j == self.grid[1] - 1)
        ]
        return

    def _set_variables(self):
        self.x = pulp.LpVariable.dicts('x', self.cells, cat=pulp.LpBinary)
        self.y = pulp.LpVariable.dicts('y', self.cells, cat=pulp.LpBinary)
        return

    def _set_objective(self):
        self.m += pulp.lpSum(self.y[(i, j)] for (i, j) in self.cells)
        return

    def _set_constraints(self):
        for (i, j) in self.cells:
            self.m += (self.x[(i, j)] + self.y[(i, j)] <= 1, f'assign-{i}-{j}')
        for (i, j) in self.cells:
            for (k, l) in self.neighbors[(i, j)]:
                self.m += (self.x[(k, l)] + self.y[(k, l)] >= self.x[(i, j)],
                           f'neighbor1-{i}-{j}-{k}-{l}')
        for (i, j) in self.boundary:
            self.m += (self.x[(i, j)] == 0, f'boundary-{i}-{j}')
        self.m += (pulp.lpSum(self.x[(i, j)] for (i,j) in self.cells) == self.num_go, f'num-go')
        return

    def _optimize(self):
        time_limit_in_seconds = 10 * 60
        self.m.solve(PULP_CBC_CMD(timeLimit=time_limit_in_seconds,
                                  gapRel=0.01))
        return

    def _is_feasible(self):
        return True

    def _process_infeasible_case(self):
        return list(), list()

    def _post_process(self):
        x_result = list()
        y_result = list()
        for (i, j) in self.cells:
            if self.x[(i, j)].value() > 0.9:
                x_result.append((i, j))
            if self.y[(i, j)].value() > 0.9:
                y_result.append((i, j))
        return x_result, y_result


def _generate_neighbors(i, j, grid):
    result = list()
    if i != 0:
        result.append((i - 1, j))
    if i != grid[0] - 1:
        result.append((i + 1, j))
    if j != 0:
        result.append((i, j - 1))
    if j != grid[1] - 1:
        result.append((i, j + 1))
    return result
