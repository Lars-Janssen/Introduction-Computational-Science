import ca
from pyics import Model
from matplotlib import pyplot
import numpy as np
import time
import sys

"""Make the simulation and set some parameters for testing."""
sim = ca.CASim()
sim.__init__()
sim.r = 1
sim.width = 50
sim.height = 50
sim.spread = 0
sim.density = 0
sim.seed = 0
sim.wind_speed = 1
sim.wind_dir = "N"

x_points = []
y_points = []

"""Samples is the amount of different lambdas checked for one seed, while
seeds is the number of seeds to be checked."""
samples = 1
seeds = 1

lines = False

print("How many seeds do you want to test?")
seeds = int(input())
print()
print("How many densities do you want to test per seed?")
samples = int(input())
print()
print("Show lines? (True or False)")
lines = input()

width = 1 if lines == "True" else 0

pyplot.suptitle("The burn time for " + str(samples * seeds) + " CA's")
for k in range(seeds):
    start_time = time.time()
    x_points = []
    y_points = []
    sim.seed = k
    for i in range(samples):
        """Set the density for even spacing."""
        sim.density = (i + 1) / samples
        sim.reset()
        while sim.done == False:
            sim.step()
        x_points.append(sim.density)
        """Append the right burn fraction to the y_points."""
        print(sim.density, sim.fraction)
        y_points.append(sim.fraction)

        """Plot the seed"""
        pyplot.plot(x_points,y_points, marker = 'o', markersize = 2,
                    linewidth=width)
    print( "%s, %s sec" % (k,(time.time() - start_time)))





pyplot.ylabel("burn fraction")
pyplot.xlabel("density")


pyplot.show()

