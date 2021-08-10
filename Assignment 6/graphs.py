###############################################################################
#                                                                             #
#                              GRAPHS ASSIGNMENT                              #
#                      INTRODUCTION COMPUTATIONAL SCIENCE                     #
#                                    2021                                     #
#                                                                             #
###############################################################################
#                                SCRIPT DETAILS                               #
###############################################################################
# This script accompanies the Graphs Jupyter Notebook and is used for         #
# functions that are a little complex for a Notebook, or require an execution #
# time that is not suitable for a Notebook.                                   #
#                                                                             #
# Run python3.8 graphs.py -h for a full description of options to run this    #
# script.                                                                     #
#                                                                             #
# You are free (and even encouraged) to write your own helper functions in    #
# this file. Just make sure that you keep the provided functions (with the    #
# same parameters).                                                           #
###############################################################################
#                               STUDENT DETAILS                               #
###############################################################################
# Name:     STUDENT NAME HERE                                                 #
# UvANetID: STUDENT NUMBER HERE                                               #
###############################################################################
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Use this PRNG to generate random numbers
rng = np.random.default_rng()


def example_graph() -> None:
    """This is an example of how to create a random graph with N = 10⁵ and
    <k> = 1 and print all of the first nodes its neighbors.

    For more information, take a look at the NetworkX documentation:
    https://networkx.org/documentation/stable/reference/
    """
    # Create a graph  with N = 10⁵  and <k> = 1.0
    G = nx.fast_gnp_random_graph(10 ** 5, 1.0 / 10 ** 5)

    # Print all neighbors of the first node
    for neighbor in G.neighbors(0):
        print(neighbor)


def scalefree_graph(N: int, avg_k: float, gamma: float = 2.5) -> nx.Graph:
    """The package NetworkX does not have a method to directly create a
    scale-free graph itself, but we can create our own using this function.
    It draws a number of degrees from the power law distribution and uses them
    to create a random graph.

    :param N:     Amount of nodes in the graph
    :param avg_k: Average degree in the graph
    :param gamma: Exponent for power law sequence
    :return:      The scale-free graph
    """
    # Generate sequence of expected degrees, one for each vertex, with
    # p(k) ~ k^(-γ)
    degrees = nx.utils.powerlaw_sequence(N, gamma)
    # rescale the expected degrees to match the given average degree
    degrees = degrees / np.mean(degrees) * avg_k

    return nx.expected_degree_graph(degrees, selfloops=False)


#########################
# HELPER FUNCTIONS HERE #
#########################


def sand_avalanche(time_steps: int, sand_lost: float,
                   scalefree: bool = False) -> np.ndarray:
    """This function should run the simulation described in section 1.3 using a
    random network with N = 10³ and an average <k> = 2.

    :param time_steps: The amount of time steps to run this simulation.
    :param sand_lost:  The fraction of sand grains lost in transfer each time
                       step.
    :param scalefree:  If true a scale-free graph should be used, if false a
                       random graph should be used. Can be ignored until 1.3b.
    :return:           A 1D numpy array containing the avalanches per time
                       step.
    """

    """Makes the graphs"""
    if(scalefree == True):
        G = scalefree_graph(10 ** 3, 2)
    else:
        G = nx.fast_gnp_random_graph(10 ** 3, 2 / 10 ** 3)

    sand_piles = [0 for i in range(nx.number_of_nodes(G))]
    array = []

    for t in range(time_steps):
        avalanches = 0
        """Choose a bucket and add a grain."""
        bucket = np.random.randint(0, nx.number_of_nodes(G))
        sand_piles[bucket] += 1
        new_buckets = [bucket]
        toppled = []

        """Dont stop until all sand is settled."""
        while len(new_buckets) != 0:
            checking = new_buckets[0]
            neighbors = list(G.neighbors(checking))
            untoppled = [x for x in neighbors if x not in toppled]
            """If the sand if higher than the neighbors, topple over to
            untoppled neighbors."""
            if sand_piles[checking] >= len(neighbors) and len(untoppled) > 0:
                toppled.append(checking)
                avalanches += 1
                falling_sand = (sand_piles[checking] / len(neighbors)) - sand_lost
                sand_piles[checking] = 0
                """Add the untoppled neighbors to check."""
                for i in list(untoppled):
                    sand_piles[i] += falling_sand
                    new_buckets.append(i)
            new_buckets.pop(0)

        array.append(avalanches)

    return array


def plot_avalanche_distribution(scalefree: bool, show: bool = False) -> None:
    """This function should run the simulation described in section 1.3 and
    plot an avalanche distribution based on the data retrieved.

    :param scalefree: If true a scale-free network is used, otherwise a random
                      network is used.
    :param show:      If true the plot is also shown in addition to being
                      stored as png.
    """
    fig = plt.figure(figsize=(7, 5))

    x = [i for i in range(10 ** 4)]
    y = sand_avalanche(10 ** 4, 10 ** -4, scalefree=scalefree)

    plt.xlabel("timestep")
    plt.ylabel("avalanches")
    if scalefree == True:
        plt.title("Avalanches for a scalefree graph")
    else:
        plt.title("Avalanches for a random graph")
    plt.plot(x,y)

    # Use different filename if random or scale-free is used.
    fig.savefig(f"1-3{'b' if scalefree else 'a'}.png")
    if show:
        plt.show()


#########################
# HELPER FUNCTIONS HERE #
#########################



def susceptible_infected(N: int, avg_k: float, i: float, time_steps: int,
                         scalefree: bool = False, avg_degree: bool = False,
                         start_infected: float = 0.001) -> np.ndarray:
    """This function should run the simulation described in section 2.1 using a
    random network.

    :param N:              The amount of nodes in the network.
    :param avg_k:          The average degree of the nodes.
    :param i:              The probability that a node will infect a neighbor.
    :param time_steps:     The amount of time steps his simulate.
    :param scalefree:      If true a scale-free network should be used, if
                           false a random network should be used. Can be
                           ignored until question 2.4a.
    :param avg_degree:     If true the average degree of the infected nodes
                           per time step should be returned instead of the
                           amount of infected nodes. Can be ignored until
                           question 2.4c.
    :param start_infected: Fraction of nodes that will be infected from the
                           start.
    :return:               1D numpy array containing the amount of infected
                           nodes per time step. (So not normalised.)
    """
    """Make the graph and infect random people."""
    G = nx.fast_gnp_random_graph(N, avg_k / N)
    spreading = list(np.random.choice(N, int(start_infected * N)))
    array = []
    infected = spreading

    for t in range(time_steps):
        array.append(len(infected))
        new_spreading = []

        """Loop over all infected nodes with uninfected neighbor."""
        for j in spreading:
            uninfected_neighbours = False
            neighbors = G.neighbors(j)
            """Look if an uninfected neighbor is infected, and if so
            add it to the spreaders for the next timestep."""
            for k in neighbors:
                if k not in infected:
                    if np.random.random() < i:
                        infected.append(k)
                        new_spreading.append(k)
                    else:
                        uninfected_neighbours = True
            """If there are still uninfected neighbours, the node is still
            a spreader."""
            if uninfected_neighbours == True:
                new_spreading.append(j)

        spreading = new_spreading
    return array


def plot_normalised_prevalence_random(start: bool, show: bool = False) -> None:
    """This function should run the simulation described in section 2.1 with
    the settings described in 2.1 b. It should then produce a plot with the
    normalised prevalence over time for the two different settings. Make sure
    to average your result over multiple runs.

    :param start: If true the simulation should be run for the first 50 time
                  steps, if false it should be run for 500 time steps.

    :param show:  If true the plot is also shown in addition to being stored
                  as png.
    """
    fig = plt.figure(figsize=(7, 5))

    if start == True:
        trials = 10
        x = np.arange(50)

        y1 = [susceptible_infected(10 ** 5, 5, 0.01, 50) for i in range(trials)]
        y2 = [susceptible_infected(10 ** 5, 0.8, 0.1, 50) for i in range(trials)]

        plt.title("The portion of the population infected averaged over " + str(trials) + " trails.")
    else:
        trials = 3
        x = np.arange(100)

        y1 = [susceptible_infected(10 ** 5, 5, 0.01, 100) for i in range(trials)]
        y2 = [susceptible_infected(10 ** 5, 0.8, 0.1, 100) for i in range(trials)]

        plt.title("The portion of the population infected averaged over " + str(trials) + " trails.")

    """Normalise the arrays."""
    for i in range(trials):
        for j in x:
            y1[i][j] /= 10 ** 5
            y2[i][j] /= 10 ** 5

    """Get the averages and standard deviations and make them 1D arrays."""
    avg_y1 = np.average(y1, axis = 0)
    avg_y2 = np.average(y2, axis = 0)
    std_y1 = np.std(y1, axis = 0)
    std_y2 = np.std(y2, axis = 0)

    avg_y1 = avg_y1.reshape(-1)
    avg_y2 = avg_y2.reshape(-1)
    std_y1 = std_y1.reshape(-1)
    std_y2 = std_y2.reshape(-1)

    """Plot and make error bars."""
    plt.plot(x, avg_y1)
    plt.fill_between(x, avg_y1 - std_y1, avg_y1 + std_y1, alpha = 0.4)
    plt.plot(x, avg_y2)
    plt.fill_between(x, avg_y2 - std_y2, avg_y2 + std_y2, alpha = 0.4)

    plt.xlabel("timestep")
    plt.ylabel("portion infected")
    plt.legend(["case (i)", "case (ii)"])

    fig.savefig(f"2-1b-{'start' if start else 'full'}.png")
    if show:
        plt.show()


def plot_approximate_R0(show: bool = False) -> None:
    """This function should run the simulation described in section 2.1 with
    the settings described in 2.1 b. It should then produce a plot with the
    approximate R0 over time for the two different settings. Make sure
    to average your result over multiple runs.

    :param show: If true the plot is also shown in addition to being stored
                 as png.
    """
    trails = 100
    R0 = [[] for i in range(trails)]
    R1 = [[] for i in range(trails)]
    fig = plt.figure(figsize=(7, 5))

    """Simulate trial times."""
    for i in range(trails):
        data = susceptible_infected(10 ** 5, 5, 0.01, 50)
        data1 = susceptible_infected(10 ** 5, 0.8, 0.1, 50)

        """Calculate the R values"""
        for t in range(len(data) - 1):
            R0[i].append((data[t+1] - data[t]) / data[t])
            R1[i].append((data1[t+1] - data1[t]) / data1[t])

    """Get the average, standard deviation and normalise."""
    avg_R0 = np.average(R0, axis = 0)
    std_R0 = np.std(R0, axis = 0)
    std_R0 = std_R0.reshape(-1)
    avg_R0 = avg_R0.reshape(-1)

    avg_R1 = np.average(R1, axis = 0)
    std_R1 = np.std(R1, axis = 0)
    std_R1 = std_R1.reshape(-1)
    avg_R1 = avg_R1.reshape(-1)

    """Plot"""
    x = np.arange(49)
    plt.plot(x,avg_R0)
    plt.fill_between(x, avg_R0 - std_R0, avg_R0 + std_R0, alpha = 0.4)
    plt.plot(x,avg_R1)
    plt.fill_between(x, avg_R1 - std_R1, avg_R1 + std_R1, alpha = 0.4)
    plt.title("The average R0 for " + str(trails) + " trails")
    plt.xlabel("timestep")
    plt.ylabel("R0")
    plt.legend(["case(i)", "case(ii)"])

    fig.savefig("2-1h.png")
    if show:
        plt.show()


def euler(f1: callable, f2: callable, I0: float, S0: float, t0: float,
          t_end: float, dt: float) -> tuple:
    """Euler method to estimate I(t) and S(t) given f1(I, S) and f2(I, S)
    with initial value I0 and S0 from t=t0, until t=t_end with a time step of
    dt.
    :param f1:    dI/dt, callable as f1(I, S).
    :param f2:    dS/dt, callable as f2(I, S).
    :param I0:    Initial value of I(t0).
    :param S0:    Initial value of S(t0).
    :param t0:    First time step to calculate.
    :param t_end: Last time step to calculate.
    :param dt:    Time between time steps.
    :return:      3 numpy arrays, corresponding to t, I(t) and S(t)
                  respectively.
    """
    t = np.arange(t0, t_end, dt)
    I = []
    S = []

    # YOUR CODE HERE

    return t, np.array(I), np.array(S)


def approximate_ODE(case: int, t0: float, t_end: float, dt: float) -> tuple:
    """Should run the euler function to approximate the ODE described in
    section 2.2 with the settings described in 2.1 b."""
    if case == 1:
        b = ...  # enter your solution of b in case (i) here
    elif case == 2:
        b = ...  # enter your solution of b in case (ii) here
    else:
        raise ValueError(f"Case must be 1 or 2, not {case}")

    # ODE functions described in section 2.2
    def f1(I, S):
        return (1 - (1 - b) ** I) * S

    def f2(I, S):
        return -(1 - (1 - b) ** I) * S

    # Approximate the solution of the ODE using Euler
    t, I, _S = ...  # YOUR CODE HERE

    return t, I


def plot_compare_ode_simulation(show: bool = False) -> None:
    """This function should run the simulation described in section 2.1 with
    the settings described in 2.1 b and approximate the same result using the
    ODE described in section 2.2 and the Euler function. It should then produce
    a plot with all four curves in one figure.

    :param show: If true the plot is also shown in addition to being stored
                 as png.
    """
    fig = plt.figure(figsize=(7, 5))

    # YOUR PLOTTING CODE HERE

    fig.savefig("2-3a.png")
    if show:
        plt.show()


def plot_normalised_prevalence_scalefree(show: bool = False) -> None:
    """This function should run the simulation described in section 2.1 with
    the settings of the case for which a giant component exists described in
    2.1 b. The simulation should be repeated using both a random and scale-free
    network. It should then produce a plot with the normalised prevalence over
    time for the two different networks. Make sure to average your result over
    multiple runs.

    :param show:  If true the plot is also shown in addition to being stored
                  as png.
    """
    fig = plt.figure(figsize=(7, 5))

    # YOUR PLOTTING CODE HERE

    fig.savefig(f"2-4a.png")
    if show:
        plt.show()


def plot_average_degree(show: bool = False) -> None:
    """This function should run the simulation described in section 2.1 with
    the settings of the case for which a giant component exists described in
    2.1 b and keep track of the average degrees of the newly infected nodes at
    every time step. The simulation should be repeated using both a random and
    scale-free network. It should then produce a plot with the average degree
    over time for the two different networks. Make sure to average your result
    over multiple runs.

    :param show:  If true the plot is also shown in addition to being stored
                  as png.
    """
    fig = plt.figure(figsize=(7, 5))

    # YOUR PLOTTING CODE HERE

    fig.savefig(f"2-4c.png")
    if show:
        plt.show()


if __name__ == '__main__':
    import argparse
    from argparse import RawTextHelpFormatter
    from sys import stderr

    # Handle command-line arguments
    assignments = ['1.3', '1.3a', '1.3b', '2.1', '2.1b', '2.1h',
                   '2.3', '2.3a', '2.4', '2.4a', '2.4c']
    description = "Run simulations and generate plots for graphs exercise."

    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter,
                                     description=description)
    parser.add_argument('-s', '--show', action='store_true',
                        help="Also show the plots in a GUI in addition to "
                             "storing them as png.")
    parser.add_argument('assignments', type=str, nargs='*',
                        choices=assignments, metavar='ASSIGNMENT',
                        help="Assignments to execute. If left out, "
                             "all assignments will be executed.\n"
                             "Choices are:\n" +
                             '\n'.join([f"'{a}'" for a in assignments]))

    args = parser.parse_args()

    if not args.assignments:
        parser.print_usage(stderr)
        print(f"Run {parser.prog} -h for more info", file=stderr)
        print("\nRunning all assignments...", file=stderr)

    # Run selected assignments
    # Plot random avalanche distribution
    if not args.assignments or '1.3' in args.assignments \
            or '1.3a' in args.assignments:
        print("Running assignment 1.3a", file=stderr)
        plot_avalanche_distribution(False, show=args.show)

    # Plot scale-free avalanche distribution
    if not args.assignments or '1.3' in args.assignments \
            or '1.3b' in args.assignments:
        print("Running assignment 1.3b", file=stderr)
        plot_avalanche_distribution(True, show=args.show)

    # Plot normalised prevalence
    if not args.assignments or '2.1' in args.assignments \
            or '2.1b' in args.assignments:
        print("Running assignment 2.1b", file=stderr)
        plot_normalised_prevalence_random(True, show=args.show)
        plot_normalised_prevalence_random(False, show=args.show)

    # Plot approximated R0
    if not args.assignments or '2.1' in args.assignments \
            or '2.1h' in args.assignments:
        print("Running assignment 2.1h", file=stderr)
        plot_approximate_R0(show=args.show)

    # Plot comparison between simulation and ODE
    if not args.assignments or '2.3' in args.assignments \
            or '2.3a' in args.assignments:
        print("Running assignment 2.3a", file=stderr)
        plot_compare_ode_simulation(show=args.show)

    # Plot comparison between scale-free and random network
    if not args.assignments or '2.4' in args.assignments \
            or '2.4a' in args.assignments:
        print("Running assignment 2.4a", file=stderr)
        plot_normalised_prevalence_scalefree(show=args.show)

    # Plot comparison between scale-free and random network
    if not args.assignments or '2.4' in args.assignments \
            or '2.4c' in args.assignments:
        print("Running assignment 2.4c", file=stderr)
        plot_average_degree(show=args.show)
