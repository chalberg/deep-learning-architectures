import context
from activation_functions.base import ActivationFunction

class Signum(ActivationFunction):
    """
    AKA: Bipolar step function
    theshold: a scalar about which the step function is centered
    """
    def __init__(self, threshold:float=0.0):
        self.threshold = threshold

    def run(self, x:float):
        if x <= self.threshold:
            return -1
        else:
            return 1