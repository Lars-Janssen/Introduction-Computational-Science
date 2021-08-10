import numpy as np
from pyics import Model
from random import randrange

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

        self.visited = []
        self.current = ""
        self.cycle_found = False

        self.make_param('r', 1)
        self.make_param('k', 2)
        self.make_param('width', 50)
        self.make_param('height', 50)
        self.make_param('rule', 30, setter=self.setter_rule)
        self.make_param('random_seed', False, bool)
        self.make_param('seed', 0)
        self.make_param('cycle_check', False, bool)
        self.make_param('cycle_stop', False, bool)

    def setter_rule(self, val):
        """Setter for the rule parameter, clipping its value between 0 and the
        maximum possible rule number."""
        rule_set_size = self.k ** (2 * self.r + 1)
        max_rule_number = self.k ** rule_set_size
        return max(0, min(val, max_rule_number - 1))

    def build_rule_set(self):
        """Sets the rule set for the current rule.
        A rule set is a list with the new state for every old configuration.

        For example, for rule=34, k=3, r=1 this function should set rule_set to
        [0, ..., 0, 1, 0, 2, 1] (length 27). This means that for example
        [2, 2, 2] -> 0 and [0, 0, 1] -> 2."""
        rule = decimal_to_base_k(self.rule, self.k)
        rule_set = [0] * (int(self.k ** (2 * self.r + 1)) - len(rule))
        rule_set.extend(rule)

        self.rule_set = rule_set

    def check_rule(self, inp):
        """Returns the new state based on the input states.

        The input state will be an array of 2r+1 items between 0 and k, the
        neighbourhood which the state of the new cell depends on."""

        """This calculates the rule that should be used. It first calculates
        the decimal equivalent of the input and then checks what the next
        symbol should be."""
        rule = 0
        for i in range(len(inp)):
            rule += int(inp[i]) * int(self.k ** (len(inp) - 1 - i))
        new = self.rule_set[len(self.rule_set) - 1 - rule]

        """If the option is selected, this adds the row to the visited rows
        and checks if the row has already been visited."""
        if self.cycle_check:
            self.current += str(new)
            if len(self.current) == self.width:
                if int(self.current) in self.visited:
                    self.cycle_found = True
                self.visited.append(int(self.current))
                self.current = ""
        return new

    def setup_initial_row(self):
        """Returns an array of length `width' with the initial state for each of
        the cells in the first row. Values should be between 0 and k."""

        """Makes a random row if the option is selected. A seed for this
        can be set, which will always give the same random row.
        If this is not chosen, a row with a dot with value k in the middle
        will be made instead. It then adds the row to the array of
        visited rows if we are checking for cycles."""
        self.current = ""
        if self.random_seed:
            np.random.seed(self.seed)
            initial = [np.random.randint(self.k) for i in range(self.width)]
        else:
            initial = [0] * self.width
            initial[self.width // 2] = self.k - 1

        if self.cycle_check:
            for i in range(self.width):
                self.current += str(initial[i])
            self.visited.append(int(self.current))
            self.current = ""
        return initial

    def reset(self):
        """Initializes the configuration of the cells and converts the entered
        rule number to a rule set."""

        self.t = 0
        self.visited = []
        self.cycle_found = False
        self.current = ""
        self.config = np.zeros([self.height, self.width])
        self.config[0, :] = self.setup_initial_row()
        self.build_rule_set()

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

        """If a cycle is found, we calculate how long it is and print
        the length."""
        if self.cycle_found and self.cycle_check:
            loop_begin = 0
            loop_end = len(self.visited) - 1
            for i in range(len(self.visited)):
                if self.visited[i] == self.visited[loop_end]:
                    loop_begin = i
                    break
            print("The length of the cycle of rule " + str(self.rule) + ": "
            + str(loop_end - loop_begin))
            self.cycle_check = False

        if self.cycle_found and self.cycle_stop:
            return True

        self.t += 1
        if self.t >= self.height:
            print("Rule " + str(self.rule) + " has no cycle in the given height.")
            return True

        for patch in range(self.width):
            # We want the items r to the left and to the right of this patch,
            # while wrapping around (e.g. index -1 is the last item on the row).
            # Since slices do not support this, we create an array with the
            # indices we want and use that to index our grid.
            indices = [i % self.width
                    for i in range(patch - self.r, patch + self.r + 1)]
            values = self.config[self.t - 1, indices]
            self.config[self.t, patch] = self.check_rule(values)


if __name__ == '__main__':
    sim = CASim()
    from pyics import GUI
    cx = GUI(sim)
    cx.start()
