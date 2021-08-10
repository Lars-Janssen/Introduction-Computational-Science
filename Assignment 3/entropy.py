import ca
from pyics import Model
from matplotlib import pyplot
import numpy as np
import time
import sys

"""Make the simulation and set some parameters for testing."""
sim = ca.CASim()
sim.__init__()
sim.r = 2
sim.k = 8
sim.width = 64
sim.height = 64
sim.random_start = True
sim.start_seed = 0
sim.use_lambda = True
sim.lambda_seed = 0
sim.build_method = 0

x_points = []
y_points = []

"""Samples is the amount of different lambdas checked for one seed, while
seeds is the number of seeds to be checked."""
samples = 1
seeds = 1

print("What do you want?")
print("0 for neighbour entropy generation")
print("1 for cell entropy generation")
print("2 for neighbour entropy from text file")
print("3 for cell entropy from text file")
mode = int(input())

lines = False
if mode == 0 or mode == 1:
    print()
    print("How many seeds do you want to test?")
    seeds = int(input())
    print("How many lambdas do you want to test per seed?")
    samples = int(input())
    print()
    print("Show lines? (True or False)")
    lines = input()

width = 1 if lines == "True" else 0


if mode == 0:
    sim.entropy_neighbour = True
    pyplot.suptitle("The average rule entropy for "
                    + str(samples * seeds) + " CA's")
    #f = open("cell_100.txt", "a")
elif mode == 1:
    sim.entropy_cell = True
    pyplot.suptitle("The average cell entropy for "
                    + str(samples * seeds) + " CA's")
    #f = open("cell_100.txt", "a")
elif mode == 2:
    """Opens the file and reads it. Then makes a plot of the points."""
    f = open("neighbour_100.txt", "r")
    for i in range(10000):
        newline = f.readline().split()
        x_points.append(float(newline[0]))
        y_points.append(float(newline[1]))
    f.close()
    pyplot.plot(x_points,y_points, marker = 'o', markersize = 2, linewidth=width)
    pyplot.suptitle("The average rule entropy for 10000 CA's")
elif mode == 3:
    """Opens the file and reads it. Then makes a plot of the points."""
    f = open("cell_100.txt", "r")
    for i in range(10000):
        newline = f.readline().split()
        x_points.append(float(newline[0]))
        y_points.append(float(newline[1]))
    f.close()
    pyplot.plot(x_points,y_points, marker = 'o', markersize = 2, linewidth=width)
    pyplot.suptitle("The average cell entropy for 10000 CA's")

if mode == 0 or mode == 1:
    for k in range(seeds):
        start_time = time.time()
        x_points = []
        y_points = []
        sim.start_seed = k
        sim.lambda_seed = k
        for i in range(samples):
            """Set the lambda for even spacing."""
            sim.lambda_param = i * (1-1/sim.k)/samples
            sim.reset()
            for j in range(sim.height):
                sim.step()
            x_points.append(sim.lambda_param)
            """Append the right entropy to the y_points."""
            if mode == 0:
                y_points.append(sim.average_entropy)
                #print(sim.lambda_param, sim.average_entropy, file=f)
            else:
                y_points.append(sim.average_cell)

            """Plot the seed"""
            pyplot.plot(x_points,y_points, marker = 'o', markersize = 2,
                        linewidth=width)
                #print(sim.lambda_param, sim.average_cell, file=f)
        print( "%s, %s sec" % (k,(time.time() - start_time)))
    #f.close()





pyplot.ylabel("average entropy")
pyplot.xlabel("lambda")


pyplot.show()

