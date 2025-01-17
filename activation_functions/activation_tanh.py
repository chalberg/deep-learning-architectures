import context
import numpy as np
from activation_functions.base import ActivationFunction

class Tanh(ActivationFunction):
    """
    Hyperbolic Tangent
    tanh(x) = (e^x - e^-x) / (e^x + e^-x)
    """
    def run(self, x:float):
        return self.f(x)
    
    def f(self, x:float):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def derivative(self, x:float):
        t_squared = self.f(x)**2
        return 1 - t_squared