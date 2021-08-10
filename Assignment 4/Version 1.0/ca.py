"""This program simulates a 1D cellular automaton. Visit
https://en.wikipedia.org/wiki/Elementary_cellular_automaton for more info.
Unfortunately, I failed to implement the isotropy condition from the paper,
as I don't know what it means."""

import numpy as np
from pyics import Model
from rule_func import *


def decimal_to_base_k(n, k):
    """Converts a given decimal (i.e. base-10 integer) to a list containing the
    base-k equivalant.

    For example, for n=34 and k=3 this function should return [1, 0, 2, 1]."""

    """I use the algorithm from the assignment instead of the numpy function,
    because now bases above 10 work easier (though they probably will not
    be used)."""
    assert n >= 0
    new_number = []
    while (n != 0):
        remainder = n % k
        n = n // k
        new_number.append(remainder)
    new_number.reverse()
    return new_number


class CASim(Model):
    def __init__(self):
        Model.__init__(self)

        self.t = 0
        self.rule_set = []
        self.config = None
        self.next = None
        self.rules = None

        self.spread = 0.5

        self.make_param('r', 1)
        self.make_param('k', 3)
        self.make_param('width', 50)
        self.make_param('height', 50)

    def setter_rule(self, val):
        """Setter for the rule parameter, clipping its value between 0 and the
        maximum possible rule number."""
        rule_set_size = self.k ** (2 * self.r + 1)
        max_rule_number = self.k ** rule_set_size
        return max(0, min(val, max_rule_number - 1))

    def lambda_rule(self, val):
        """Setter for lambda_param, clipping it between 0 and 1-(1/k)."""
        maximum = 1-(1/self.k)
        return max(0, min(val, maximum))

    def build_rule(self, val):
        """Setter for the build_method."""
        number = round(val, 0)
        if number >= 1:
            return 1
        else:
            return 0

    def build_rules(self):
        self.rules = {
            0: empty,
            1: tree,
            2: fire
        }
        return 0

    def check_rule(self, inp):
        """Returns the new state based on the input states.

        The input state will be an array of 2r+1 items between 0 and k, the
        neighbourhood which the state of the new cell depends on."""

        """This calculates the rule that should be used. It first calculates
        the decimal equivalent of the input and then checks what the next
        symbol should be."""
        cell = inp[self.r][self.r]

        return self.rules[cell](inp, self.spread)

    def setup_initial_state(self):
        """Returns an array with the initial state for each of
        the cells in the first timestep. Values should be between 0 and k."""
        for i in range(self.width):
            if i == 0:
                for j in range(self.height):
                    self.config[j][i] = 2
            else:
                for j in range(self.height):
                    self.config[j][i] = np.random.randint(0, 2)

        return self.config

    def reset(self):
        """Initializes the configuration of the cells and converts the entered
        rule number to a rule set."""

        self.t = 0
        self.config = np.zeros([self.height, self.width])
        self.config = self.setup_initial_state()
        self.build_rules()

    def draw(self):
        """Draws the current state of the grid."""

        import matplotlib
        import matplotlib.pyplot as plt

        plt.cla()
        if not plt.gca().yaxis_inverted():
            plt.gca().invert_yaxis()
        plt.imshow(self.config, interpolation='none', vmin=0, vmax=self.k - 1,
                   cmap=matplotlib.cm.binary)
        plt.axis('image')
        plt.title('t = %d' % self.t)

    def step(self):
        """Performs a single step of the simulation by advancing time (and thus
        row) and applying the rule to determine the state of the cells."""

        self.t += 1
        if self.t >= self.height:
            return True

        length = 2 * self.r + 1
        values = np.zeros([length, length])
        self.next = np.zeros([self.height, self.width])
        for i in range(self.width):
            for j in range(self.height):
                values = neighbour_func(self, i, j)
                self.next[j][i] = self.check_rule(values)
        self.config = self.next

def neighbour_func(self, dot_x, dot_y):
    length = 2 * self.r + 1
    neighbours = np.zeros([length, length])
    for i in range(length):
        x = dot_x + i - self.r
        for j in range(length):
            y = dot_y + j - self.r
            if x >= 0 and y >= 0 and x < self.width and y < self.height:
                neighbours[j][i] = self.config[y][x]
    return neighbours




if __name__ == '__main__':
    sim = CASim()
    from pyics import GUI
    cx = GUI(sim)
    cx.start()
