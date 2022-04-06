import math

from numpy import absolute
def cost_function_absolute(actual_temperatures, estimated_temperatures):
    difference = estimated_temperatures - actual_temperatures
    cost = sum(abs(difference))

    return difference, cost
