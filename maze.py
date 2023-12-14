import numpy as np


class Maze:
    def __init__(self, size=10, start=(0, 0), goal=None, num_traps=None, random_seed=None):
        self.size = size
        self.start = start
        self.goal = goal if goal is not None else (size - 1, size - 1)
        self.grid = np.zeros((size, size))
        self.traps = self._place_traps(num_traps, random_seed)
        self.grid[self.goal] = 1


    def _place_traps(self, num_traps, random_seed):
        np.random.seed(random_seed)
        traps = set()
        while len(traps) < num_traps:
            trap = (np.random.randint(0, self.size), np.random.randint(0, self.size))
            if trap != self.start and trap != self.goal:
                traps.add(trap)
                self.grid[trap] = -1
        return traps

    def is_goal(self, position):
        return position == self.goal

    def is_trap(self, position):
        return position in self.traps

    def get_state(self):
        state = np.zeros(self.size * self.size)
        state[self.start[0] * self.size + self.start[1]] = 1
        return state

    def get_state_o(self, position):
        return self.grid[position]

    def reset(self):
        self.start = (0, 0)
        return self.get_state()