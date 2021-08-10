"""
NAME: Aron de Ruijter, Lars Janssen
STUDENT ID: 12868655, 12882712

rules.py

This program plots the critical density.
"""

import ca
from pyics import Model
from matplotlib import pyplot
import numpy as np
import time
import sys

"""Make the simulation and set some parameters for testing."""
sim = ca.CASim()
sim.__init__()
sim.width = 100
sim.height = 100
sim.density = 1
sim.seed = 0
sim.wind_speed = 0
sim.wind_angle = 0
sim.weather = 1
sim.spread = np.array([0.2, 0.3, 0.4])
sim.firebreak = 0

x_points = []
y_points = []

"""Samples is the amount of different densities checked for one seed, while
seeds is the number of seeds to be checked."""
samples = 1
seeds = 1

lines = False

print("How many seeds do you want to test?")
seeds = int(input())
print()
print("How many densities do you want to test per seed?")
samples = int(input())

pyplot.suptitle("The burn fraction for " + str(samples * seeds) + " CA's")
for k in range(seeds):
    start_time = time.time()
    x_points = []
    y_points = []
    sim.seed = k
    for i in range(samples):
        """Set the densities for even spacing."""
        sim.density= (i + 1) / samples
        sim.reset()
        while sim.done == False:
            sim.step()
        x_points.append(sim.density)
        """Append the right burn fraction to the y_points."""
        y_points.append(sim.fraction)

        """Plot the seed"""
        pyplot.plot(x_points,y_points, marker = 'o', markersize = 2,
                    linewidth=0)
    print( "%s, %s sec" % (k,(time.time() - start_time)))





pyplot.ylabel("burn fraction")
pyplot.xlabel("density")


pyplot.show()

