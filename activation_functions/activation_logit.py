import context
import numpy as np
from activation_functions.base import ActivationFunction

class Logit(ActivationFunction):
    """
    Logistic Function
    alpha: scalar controlling the firing rate of the neuron
    logit(x) = 1 / (1 + e^{-alpha * x})
    """
    def __init__(self, alpha:float):
        self.alpha = alpha
        if self.alpha <= 0:
            raise("Parameter alpha must be positive.")
        
    def run(self, x:float):
        return self.f(x)
    
    def f(self, x:float):
        return 1 / (1 + np.exp(-self.alpha * x))

    def derivative(self, x:float):
        fx = self.f(x)
        return self.alpha*fx*(1 - fx)