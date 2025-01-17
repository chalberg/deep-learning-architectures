import context
import numpy as np
from activation_functions.base import ActivationFunction

class Arctan(ActivationFunction):
    """
    Inverse Tangent
    f(x) = 2/pi * tanh^-1(x)
    """
    def run(self, x:float):
        return (2/np.pi) * np.arctanh(x)