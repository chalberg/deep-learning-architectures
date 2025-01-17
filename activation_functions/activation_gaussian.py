import context
import numpy as np
from activation_functions.base import ActivationFunction

class Gaussian(ActivationFunction):
    """
    Gaussian
    f(x) = e^{-x^2}
    """
    def run(self, x:float):
        return self.f(x)
    
    def f(self, x:float):
        return np.exp(-(x**2))
    
    def derivative(self, x:float):
        return -2*x*self.f(x)