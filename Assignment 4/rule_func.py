"""
NAME: Aron de Ruijter, Lars Janssen
STUDENT ID: 12868655, 12882712

rule_func.py

In this file are the functions for checking if the neighbouring cells
should change state and calculating the probabilities of the fire spreading.
"""

import numpy as np
from numpy.random import random as rd


def neighbours(self, x, y):
    """This checks if a neighbouring cell should change its state to fire
    based on the parameters."""

    chance = self.chance

    """These are the positions of neighbouring cells."""
    pos = [[x, y-1], [x+1, y-1], [x+1, y], [x+1, y+1], [x, y+1], [x-1, y+1],
           [x-1, y], [x-1, y-1]]

    for i in range(len(pos)):
        """This checks if there really is a cell in the position"""
        if (pos[i][0] >= 0 and pos[i][1] >= 0 and pos[i][0] < self.width and
                pos[i][1] < self.height):
            cell_state = int(self.config[pos[i][1]][pos[i][0]])
            """Selects the neighbour can catch fire, the random number is under
            the probabilty of spreading, and we have not already selected
            it. It selects it for the next timestep and changes its state
            to burning."""
            if (cell_state > 2 and rd() < chance[i][cell_state - 3] and
                    pos[i] not in self.new_changable):
                self.next[pos[i][1]][pos[i][0]] = 2
                self.new_changable.append(pos[i])


def calc_chances(self):
    """This calculates the chance of spreading to each of the neighbouring
       cells."""

    """The angle the wind is blowing in radians"""
    base_angle = self.wind_angle * np.pi/180

    chance = [0] * 8

    """Some constants for the wind parameter."""
    C = 0.045
    c = 0.131
    V = self.wind_speed

    weather_spread = self.spread
    """This modifies the base spread chance based the weather conditions."""
    if self.weather < 1:
        weather_spread = good_weather(self.spread, self.weather)
    else:
        weather_spread = bad_weather(self.spread, self.weather)

    """This modifies the spread chance based on wind speed and direction."""
    for i in range(8):
        angle = base_angle - (i/4) * np.pi
        f = np.exp(self.wind_speed * c * (np.cos(angle) - 1))
        wind_param = f * np.exp(C * V)
        chance[i] = list(weather_spread * wind_param)
    return chance


def good_weather(p_old, h):
    """This calculates the chance of vegetation catching fire, based on the
    weather value h, when h is below 1"""
    return p_old * (1 - (1 - h)**2)


def bad_weather(p_old, h):
    """This calculates the chance of vegetation catching fire, based on the
    weather value h, when h is above 1"""
    return p_old + (1 - p_old) * (1 - (1 / h))**2
