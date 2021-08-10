import numpy as np
from numpy.random import random as rd

def empty():
    return 0

def tree(self, i, j):
    return 1

def burnt():
    return 2

def fire(self, i, j):
    self.new_changable.append([i,j])
    neighbours(self, i, j)
    return 2


def neighbours(self, x, y):
    wind = self.wind
    if self.weather < 1:
        chance = good_weather(self.spread, self.weather)
    else:
        chance = bad_weather(self.spread, self.weather)
    # chance = self.spread
    diag_chance = chance / 4
    new_fires = []
    #N
    if y > 0 and self.config[y - 1][x] == 1 and rd() < chance * wind[0]:
        self.next[y - 1][x] = 3
        new_fires.append([x, y - 1])
    #NE
    if x < self.width - 1 and y > 0 and self.config[y - 1][x + 1] == 1 and rd() < diag_chance * wind[1]:
        self.next[y - 1][x + 1] = 3
        new_fires.append([x + 1, y - 1])
    #E
    if x < self.width - 1 and self.config[y][x + 1] == 1 and rd() < chance * wind[2]:
        self.next[y][x + 1] = 3
        new_fires.append([x + 1, y])
    #SE
    if x < self.width - 1  and y < self.height - 1 and self.config[y + 1][x + 1] == 1 and rd() < diag_chance * wind[3]:
        self.next[y + 1][x + 1] = 3
        new_fires.append([x + 1, y + 1])
    #S
    if y < self.height - 1 and self.config[y + 1][x] == 1 and rd() < chance * wind[4]:
        self.next[y + 1][x] = 3
        new_fires.append([x, y + 1])
    #SW
    if x > 0  and y < self.height - 1 and self.config[y + 1][x - 1] == 1 and rd() < diag_chance * wind[5]:
        self.next[y + 1][x - 1] = 3
        new_fires.append([x - 1, y + 1])
    #W
    if x > 0 and self.config[y][x - 1] == 1 and rd() < chance * wind[6]:
        self.next[y][x - 1] = 3
        new_fires.append([x - 1, y])
    #NW
    if x > 0  and y > 0 and self.config[y - 1][x - 1] == 1 and rd() < diag_chance * wind[7]:
        self.next[y - 1][x - 1] = 3
        new_fires.append([x - 1, y - 1])

    for i in new_fires:
        if i not in self.new_changable:
            self.new_changable.append(i)

def wind_weights(dir, speed):
    wind = [0] * 8
    offset = 0
    if dir == "E":
        offset = 2
    elif dir == "S":
        offset = 4
    elif dir == "W":
        offset = 6

    norm = 1 + speed

    wind[offset % 8] = 1 + (speed * 2) / norm
    wind[(1 + offset) % 8] = 1 + speed / norm
    wind[(2 + offset) % 8] = 1
    wind[(3 + offset) % 8] = 1 - (speed * 0.5) / norm
    wind[(4 + offset) % 8] = 1 - speed / norm
    wind[(5 + offset) % 8] = 1 - (speed * 0.5) / norm
    wind[(6 + offset) % 8] = 1
    wind[(7 + offset) % 8] = 1 + speed / norm

    return wind


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