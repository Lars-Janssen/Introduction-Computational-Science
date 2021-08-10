import numpy as np

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
                               replace=False)
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
            rule += int(current_neighbours[j] *
                        self.k ** ((2 * self.r + 1) - j - 1))
        amount[len(amount) - 1 - rule] += 1
        current_neighbours.pop(0)
        current_neighbours.append(
            self.config[self.t, (2 * self.r + 1 + i) % self.width])

    """Calculates the rule for the last neighbourhood."""
    rule = 0
    for j in range(len(current_neighbours)):
        rule += int(current_neighbours[j] *
                    self.k ** ((2 * self.r + 1) - j - 1))
    amount[len(amount)-1 - rule] += 1

    """Calculates the Shannon entropy and the the average entropy so far."""
    shannon = 0
    for i in range(len(amount)):
        if(amount[i] != 0):
            probability = amount[i] / self.width
            shannon -= probability * np.log2(probability)
    self.average_entropy = (self.average_entropy *
                            self.t + shannon) / (self.t + 1)


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