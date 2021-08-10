"""
NAME: Aron de Ruijter, Lars Janssen
STUDENT ID: 12868655, 12882712

rules.py

This program creates the setters for the parameters.
"""

def density_rule(val):
    """Setter for density, clipping it between 0 and 1."""
    maximum = 1
    return max(0, min(val, maximum))

def weather_rule(val):
    """Setter for weather, clipping it between 0 and 2."""
    maximum = 2
    return max(0, min(val, maximum))

def angle_rule(val):
    """Setter for angle, clipping it between 0 and 360 by using the modulo.
    No negative angles allowed."""
    maximum = 360
    return max(0, val % 360)

def speed_rule(val):
    """Setter for wind speed, making it positive"""
    return max(0, val)

def firebreak_rule(val):
    """Setter for firebreak, making it have to be positive."""
    return max(0, val)