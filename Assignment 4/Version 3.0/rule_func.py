import numpy as np
from numpy.random import random as rd
import time

def empty():
    return 0

def tree(self, i, j):
    return 1

def burnt():
    return 2

def fire(self, i, j):
    neighbours(self, i, j)
    return 2


def neighbours(self, x, y):
    chance = self.chance
    nx = [0,1,1,1,0,-1,-1,-1]
    ny = [-1,-1,0,1,1,1,0,-1]
    positions = [[x,y-1],[x+1,y-1],[x+1,y],[x+1,y+1],[x,y+1],[x-1,y+1],[x-1,y],
                 [x-1,y-1]]

    for i in range(8):
        if rd() < chance[i]:
            if x + nx[i] >= 0 and y + ny[i] >= 0 and x + nx[i] < self.width and y + ny[i] < self.height:
                if self.config[y + ny[i]][x + nx[i]] == 1 and positions[i] not in self.new_changable:
                    self.next[positions[i][1]][positions[i][0]] = 3
                    self.new_changable.append(positions[i])


def wind_weights(self):
    base_angle = self.wind_angle * np.pi/180

    chance = [0] * 8

    C = 0.045
    c = 0.131
    V = self.wind_speed

    for i in range(8):
        angle = base_angle - (i/4) * np.pi
        f = np.exp(self.wind_speed * c * (np.cos(angle) - 1))
        wind_param = f * np.exp(C * V)
        chance[i] = self.spread * wind_param
    """     norm = 1 + speed
    diag_spread = self.spread / 4

    coefficients = [2, 1, 0, 0.5, 1, 0.5, 0, 1]
    spreads = [self.spread, diag_spread]

    for i in range(8):
        wind[(i + offset) % 8] = 1 + speed / norm
        if i < 3 or i > 5:
            wind[(i + offset) % 8] = 1 + (speed * coefficients[i]) / norm
            chance[(i + offset) % 8 ] = 1 - (1 - spreads[i % 2]) / wind[(i + offset) % 8]
        else:
            wind[(i + offset) % 8] = 1 - (speed * coefficients[i]) / norm
            chance[(i + offset) % 8 ] = spreads[i % 2] * wind[(i + offset) % 8]
    """
    return chance


def good_weather(p_old, h):
    return p_old * (1 - (1 - h)**2)


def bad_weather(p_old, h):
    return p_old + (1 - p_old) * (1 - (1 / h))**2

def calculate(self):
    for i in range(1, self.width):
        for j in range(self.height):
            if self.config[j][i] == 2:
                self.fraction += 1
    self.fraction = self.fraction / (self.width * self.height * self.density)