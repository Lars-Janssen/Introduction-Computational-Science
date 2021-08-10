import numpy as np

def empty(inp, spread):
    return 0

def tree(inp, spread):
    for i in range(len(inp[0])):
        for j in range(len(inp)):
            if inp[i][j] == 2 and np.random.random() < spread:
                return 2
    return 1

def fire(inp, spread):
    return 0