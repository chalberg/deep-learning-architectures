import context
import numpy as np
from activation_functions.base import ActivationFunction

class ReLU(ActivationFunction):
    """
    Rectified Linear Unit
    """
    def __init__(self):
        pass

    def run(self, x:float):
        return np.max(0, x)