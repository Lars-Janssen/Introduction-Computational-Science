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

"""     for i in range(1, self.r + 1):
        chance = self.spread / (2 ** (i - 1))
        if x >= i and self.config[y][x - i] == 1:
            if([x - i,y] not in self.new_changable) and rd() < chance:
                self.new_changable.append([x - i,y])
                self.next[y][x - i] = 3
        if y >= i and self.config[y - i][x] == 1:
            if([x,y - i] not in self.new_changable) and rd() < chance:
                self.new_changable.append([x,y - i])
                self.next[y - i][x] = 3
        if x < self.width - i and self.config[y][x + i] == 1:
            if([x + i,y] not in self.new_changable) and rd() < chance:
                self.new_changable.append([x + i,y])
                self.next[y][x + i] = 3
        if y < self.height - i and self.config[y + i][x] == 1:
            if([x,y + i] not in self.new_changable) and rd() < chance:
                self.new_changable.append([x,y + i])
                self.next[y + i][x] = 3 """

def neighbours(self, x, y):
    for r in range(1, self.r + 1):
        chance = self.spread / (9 ** (r - 1))
        for i in range(-r, r + 1):
            if x + i >= 0 and x + i <= self.width - 1 and y - r >= 0 and y + r <= self.height - 1:
                if  rd() < chance and self.config[y - r][x + i] == 1 and [x + i, y - r] not in self.new_changable:
                    self.new_changable.append([x + i,y - r])
                    self.next[y - r][x + i] = 3
                if rd() < chance and self.config[y + r][x + i] == 1 and [x + i, y + r] not in self.new_changable:
                    self.new_changable.append([x + i,y + r])
                    self.next[y + r][x + i] = 3

            for j in range(-r + 1, r):
                if x - r >= 0 and x + r <= self.width - 1 and y - j >= 0 and y + j <= self.height - 1:
                    if rd() < chance and self.config[y + j][x - r] == 1 and [x - r, y + j] not in self.new_changable:
                        self.new_changable.append([x - r,y + j])
                        self.next[y + j][x - r] = 3
                    if rd() < chance and self.config[y + j][x + r] == 1 and [x + r, y + j] not in self.new_changable:
                        self.new_changable.append([x + r,y + j])
                        self.next[y + j][x + r] = 3



def calculate(self):
    for i in range(1, self.width):
        for j in range(self.height):
            if self.config[j][i] == 2:
                self.fraction += 1
    self.fraction = self.fraction / (self.width * self.height * self.density)