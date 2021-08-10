"""This program simulates a 1D cellular automaton. Visit
https://en.wikipedia.org/wiki/Elementary_cellular_automaton for more info.
Unfortunately, I failed to implement the isotropy condition from the paper,
as I don't know what it means."""

import numpy as np
from pyics import Model
from random import randrange
from random import sample


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


def random_table(self):
    """Generates the rule set based on a lambda value using the random table
    method. It essentialy flips a coin with probability lambda for every rule.
    you can read more about it on page 4 of Langton's paper. Note that
    a neighbourhood of zeroes will result in zeroes, as in Langton's paper."""
    rule_set = []
    for i in range(self.k ** (2 * self.r + 1) - 1):
        g = np.random.rand()
        if g > self.lambda_param:
            g = 0
        else:
            g = np.random.randint(1, self.k)
        rule_set.append(g)
    rule_set.append(0)
    return rule_set


def table_walk(self):
    """Generates the rule set by starting with a rule set of only zeroes,
    and then changing some rules, until we achieve the required lambda. This
    way, when we increase lambda, the result will still bear some resemblance
    to the automaton with a lower lambda. Note that a neighbourhood of zeroes
    will result in zeroes, as in Langton's paper. This works as a table
    walkthrough, because choice will give the same result with the same seed,
    it will just give more rules to change with a higher lambda."""
    power = self.k ** (2 * self.r + 1)
    rule_set = [0] * power
    changes = np.random.choice(power - 1, int(self.lambda_param * power),
              replace = False)
    for i in range(len(changes)):
        rule_set[changes[i]] = np.random.randint(1, self.k)
    return rule_set


def entropy(self):
    """Calculates the entropy per rule. It counts the number of times each rule
    is in a row, and then calculates the shannon entropy. It than calculates
    the average Shannon entropy over the entire height."""

    """Gets the first neighbours, which are the first 2*r+1 cells."""
    current_neighbours = []
    amount = [0] * self.k ** (2 * self.r + 1)
    for i in range(2 * self.r + 1):
        current_neighbours.append(self.config[self.t, i % self.width])

    """Calculates the rule and adds one to it's amount. It then removes the
    leftmost cell and adds a cell to the right."""
    for i in range(len(self.config[self.t]) - 1):
        rule = 0
        for j in range(len(current_neighbours)):
            rule += int(current_neighbours[j] * self.k ** ((2 * self.r + 1) - j - 1))
        amount[len(amount)- 1 - rule] += 1
        current_neighbours.pop(0)
        current_neighbours.append(self.config[self.t, (2 * self.r + 1 + i) % self.width])

    """Calculates the rule for the last neighbourhood."""
    rule = 0
    for j in range(len(current_neighbours)):
        rule += int(current_neighbours[j] * self.k ** ((2 * self.r + 1) - j - 1))
    amount[len(amount)-1 - rule] += 1

    """Calculates the Shannon entropy and the the average entropy so far."""
    shannon = 0
    for i in range(len(amount)):
        if(amount[i] != 0):
            probability = amount[i] / self.width
            shannon -= probability * np.log2(probability)
    self.average_entropy = (self.average_entropy * self.t + shannon) / (self.t + 1)


def entropycell(self):
    """Calculates the entropy per cell by first calculating how often each
    state appears in a row, and then calculating the Shannon entropy."""
    cells = [0] * self.k
    for i in range(self.width):
        cells[int(self.config[self.t, i])] += 1

    """Calculates the Shannon entropy and the the average entropy so far."""
    shannon = 0
    for i in range(self.k):
        if(cells[i] != 0):
            probability = cells[i] / self.width
            shannon -= probability * np.log2(probability)
    self.average_cell = (self.average_cell * self.t + shannon) / (self.t + 1)


class CASim(Model):
    def __init__(self):
        Model.__init__(self)

        self.t = 0
        self.rule_set = []
        self.config = None

        self.visited = []
        self.current = ""
        self.cycle_found = False

        self.average_entropy = 0
        self.average_cell = 0

        self.make_param('r', 1)
        self.make_param('k', 2)
        self.make_param('width', 50)
        self.make_param('height', 50)
        self.make_param('rule', 30, setter=self.setter_rule)
        self.make_param('random_start', False, bool)
        self.make_param('start_seed', 0)
        self.make_param('cycle_check', False, bool)
        self.make_param('cycle_stop', False, bool)
        #If lambda is used, rule will not be used.
        self.make_param('use_lambda', False, bool)
        self.make_param('lambda_seed', 0)
        self.make_param('lambda_param', 0, float, setter=self.lambda_rule)
        #0 for table walk, 1 for random table
        self.make_param('build_method', 0, setter=self.build_rule)
        #Set to True to calculate the entropy
        self.make_param('entropy_neighbour', False, bool)
        self.make_param('entropy_cell', False, bool)

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

    def build_rule_set(self):
        """Sets the rule set for the current rule.
        A rule set is a list with the new state for every old configuration.

        For example, for rule=34, k=3, r=1 this function should set rule_set to
        [0, ..., 0, 1, 0, 2, 1] (length 27). This means that for example
        [2, 2, 2] -> 0 and [0, 0, 1] -> 2."""
        if self.use_lambda == False:
            rule = decimal_to_base_k(self.rule, self.k)
            rule_set = [0] * (int(self.k ** (2 * self.r + 1)) - len(rule))
            rule_set.extend(rule)
            self.rule_set = rule_set
        else:
            """Generate the rule set using the chosen build method."""
            np.random.seed(self.lambda_seed)
            if self.build_method == 0:
                self. rule_set = table_walk(self)

            if self.build_method == 1:
                self.rule_set = random_table(self)

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
        if self.random_start:
            np.random.seed(self.start_seed)
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

        """Calculate the chosen entropies"""
        if self.entropy_neighbour:
            entropy(self)

        if self.entropy_cell:
            entropycell(self)

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
