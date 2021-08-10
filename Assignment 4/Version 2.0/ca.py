"""This program simulates a 1D cellular automaton. Visit
https://en.wikipedia.org/wiki/Elementary_cellular_automaton for more info.
Unfortunately, I failed to implement the isotropy condition from the paper,
as I don't know what it means."""

import numpy as np
from pyics import Model
from rule_func import *
from copy import deepcopy


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
        self.changable = []
        self.new_changable = []
        self.rules = None
        self.k = 4
        self.done = False
        self.fraction = 0


        self.make_param('r', 1)
        self.make_param('width', 50)
        self.make_param('height', 50)
        self.make_param('spread', 0.5)
        self.make_param('density', 0.5)
        self.make_param('seed', 0)

    def build_rules(self):
        self.rules = {
            0: empty,
            1: tree,
            2: burnt,
            3: fire
        }
        return 0

    def check_rule(self, i, j):
        """Returns the new state based on the input states.

        The input state will be an array of 2r+1 items between 0 and k, the
        neighbourhood which the state of the new cell depends on."""

        """This calculates the rule that should be used. It first calculates
        the decimal equivalent of the input and then checks what the next
        symbol should be."""
        cell_state = self.config[j][i]
        if cell_state == 1 or cell_state == 3:
            return self.rules[cell_state](self, i, j)
        else:
            return self.rules[cell_state]()

    def setup_initial_state(self):
        """Returns an array with the initial state for each of
        the cells in the first timestep. Values should be between 0 and k."""
        np.random.seed(self.seed)

        one_amount = int((self.width * self.height) * self.density)
        self.config[:one_amount] = 1
        np.random.shuffle(self.config)
        self.config = self.config.reshape((self.height, self.width))

        for j in range(self.height):
            self.config[j][0] = 3
            self.changable.append([0,j])
            for i in range(3):
                self.config[j][self.width // 2 + i] = 0

        return self.config

    def reset(self):
        """Initializes the configuration of the cells and converts the entered
        rule number to a rule set."""

        self.t = 0
        self.config = np.zeros(self.height * self.width)
        self.changable = []
        self.config = self.setup_initial_state()
        self.build_rules()
        self.done = False
        self.fraction = 0

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
        self.new_changable = []
        same = True
        length = 2 * self.r + 1
        self.next = deepcopy(self.config)
        for k in range(len(self.changable)):
            i = self.changable[k][0]
            j = self.changable[k][1]
            self.next[j][i] = self.check_rule(i, j)
            if same == True and self.next[j][i] != self.config[j][i]:
                    same = False
        if same == True:
            self.done = True
            calculate(self)
            return True
        self.config = self.next
        self.changable = self.new_changable






if __name__ == '__main__':
    sim = CASim()
    from pyics import GUI
    cx = GUI(sim)
    cx.start()
