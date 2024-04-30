import math
from pprint import pprint

from model.go_qi import GoQi
from output_handler.drawer import draw_solution


def main():
    for num_go in range(1, 30):
        length = math.ceil(math.sqrt(num_go * 2) + 2)
        grid = (length, length)

        model = GoQi(grid, num_go)
        x_result, y_result = model.solve()
        draw_solution(grid, x_result, y_result,
                      f'data/output/go_{num_go}_{len(y_result)}')
    return


if __name__ == '__main__':
    main()
