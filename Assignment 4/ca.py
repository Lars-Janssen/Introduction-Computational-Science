"""
NAME: Aron de Ruijter, Lars Janssen
STUDENT ID: 12868655, 12882712

ca.py

This program simulates a 2D cellular automaton of a forest fire. Visit
https://en.wikipedia.org/wiki/Elementary_cellular_automaton for more info.
We use the wind direction, wind speed, weather and different densities to
get a realistic model.
"""

import numpy as np
from pyics import Model
from rule_func import *
from copy import deepcopy
from rules import *


class CASim(Model):
    def __init__(self):
        Model.__init__(self)

        self.t = 0

        self.config = None
        self.next = None
        self.changable = []
        self.new_changable = []

        self.states = 7
        self.done = False
        self.fraction = 0

        self.chance = [0] * 8
        self.spread = np.array([0.2, 0.3, 0.4, 0.05])

        self.make_param('width', 100)
        self.make_param('height', 100)
        self.make_param('density', 0.9, setter=density_rule)
        self.make_param('weather', 1.0, setter=weather_rule)
        self.make_param('seed', 0)
        self.make_param('wind_angle', 0.0, setter=angle_rule)
        self.make_param('wind_speed', 0.0, setter=speed_rule)
        self.make_param('firebreak', 0, setter=firebreak_rule)



    def build_rules(self):
        """This gives each of the directions the fire can spread in the appropiate
        chance based on the wind conditions"""
        self.chance = calc_chances(self)
        return 0

    def check_rule(self, i, j):
        """This checks the rule, and looks if the fire will spread."""
        neighbours(self, i, j)
        return 0

    def setup_initial_state(self):
        """Returns an array with the initial state for each of
        the cells in the first timestep. This is based on the given density."""
        np.random.seed(self.seed)

        """This chooses a number of the first cells based on the density, and
        then fills these cells with farmland, bushes, and trees, which are
        states 3,4, and 5 respectively. This is also done according to a
        distribution"""
        amount = int((self.width * self.height) * self.density)
        for i in range(amount):
            self.config[i] = np.random.choice([3, 4, 5], p=[0.25, 0.25, 0.5])

        """This randomizes the array and makes it into a square."""
        np.random.shuffle(self.config)
        self.config = self.config.reshape((self.height, self.width))

        """This sets the first collumn on fire"""
        for j in range(self.height):
            self.config[j][0] = 2
            self.changable.append([0, j])

        """This creates a firebreak."""
        for j in range(self.height):
            for i in range(self.firebreak):
                if(self.config[j][self.width // 2 + i] > 2):
                    self.config[j][self.width // 2 + i] = 6

        """This sets the center cell on fire"""
        # self.config[self.height // 2][self.width // 2] = 2
        # self.changable.append([self.width // 2, self.height // 2])

        return self.config

    def reset(self):
        """Initializes the first of the cells and calculates the spread
        chances."""
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
        import matplotlib.colors as clrs

        """Gives the states nice colors"""
        cmap = clrs.ListedColormap(["white", "black", "red", "lawngreen",
                                    "green", "darkgreen", "gray"])
        plt.cla()
        if not plt.gca().yaxis_inverted():
            plt.gca().invert_yaxis()
        plt.imshow(self.config, interpolation='none', vmin=0,
                   vmax=self.states - 1, cmap=cmap)
        plt.axis('image')
        plt.title('t = %d' % self.t)

    def step(self):
        """Performs a single step of the simulation by advancing time
        and applying the rule to determine the next state of the cells."""

        """With this you can change  wind speed and angle
        during the simulation."""
        self.t += 1

        self.new_changable = []
        self.next = deepcopy(self.config)

        """Loop over all the cells which are one fire, since only they
        determine which cells will change (also become on fire or burn up)."""
        for k in range(len(self.changable)):
            i = self.changable[k][0]
            j = self.changable[k][1]
            """Check if neighbours are ignited."""
            self.check_rule(i, j)
            """Change its own state to burnt."""
            self.next[j][i] = 1

        comparison = self.next == self.config
        if comparison.all():
            self.done = True
            calculate(self)
            return True
        self.config = self.next
        self.changable = self.new_changable


def calculate(self):
    """This calculates the percentage of the forest that has burned."""
    for i in range(1, self.width):
        for j in range(self.height):
            if self.config[j][i] == 1:
                self.fraction += 1
    self.fraction = self.fraction / (self.width * self.height * self.density)


if __name__ == '__main__':
    sim = CASim()
    from pyics import GUI
    cx = GUI(sim)
    cx.start()
