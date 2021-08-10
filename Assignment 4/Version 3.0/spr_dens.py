import ca
from pyics import Model
from matplotlib import pyplot
from matplotlib import colors
import numpy as np
import time
import sys

"""Make the simulation and set some parameters for testing."""
sim = ca.CASim()
sim.__init__()
sim.width = 50
sim.height = 50
sim.spread = 0
sim.density = 0
sim.seed = 12
sim.wind_dir = "W"
sim.wind_speed = 2

"""Samples is the amount of different lambdas checked for one seed, while
seeds is the number of seeds to be checked."""
samples = 1
seeds = 1
spreads = 1

lines = False

print("How many densities do you want to test?")
samples = int(input())
print()
print("How many spreads do you want to test?")
spreads = int(input())
c_points = np.zeros((spreads, samples))
pyplot.suptitle("The burn time for " + str(spreads * samples) + " CA's")
for i in range(samples):
    start_time = time.time()
    for j in range(spreads):
        """Set the density for even spacing."""
        sim.density = (i + 1) / samples
        sim.spread = (j + 1) / spreads
        sim.reset()
        while sim.done == False:
            sim.step()

        c_points[j][i] = sim.fraction
    print( "%s, %s sec" % (i,(time.time() - start_time)))

"""Plot"""
pyplot.imshow(c_points, interpolation='none')



pyplot.ylabel("spread")
pyplot.xlabel("density")


pyplot.show()
