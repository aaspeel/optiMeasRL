"""
The linear penalization will add a step error, defined as O.5/threshold, multiplied by the number
of measurement taken to the error made by the predictor to have a linearly increasing error.
0.5 is used because if the predictor doesn't get any measurement it should predict 0.5 as its 
default value which means that the maximum error the predictor should make is 0.5
given that the signal is between 0 and 1.
"""

from penalization import Penalization

class LinearPenalization(Penalization):

    def __init__(self, threshold):
        """
        Arguments
        ---------
        threshold : the maximum number of measures allowed (must be > 0)
        """
        self._threshold = threshold
        self._numberOfMeasure = 0
        assert(self._threshold > 0)
        self._errorStep = 0.5/self.threshold


    def reset(self):
        """
        Rest internal parameters
        """
        self._numberOfMeasure = 0


    def get(self, error, action, *args, **kwargs):
        """ Return the penalization for the RL agent given a certain criteria
        Arguments
        ---------
        error : the error made by the predictor
        action : the action taken by the agent (1 means measure taken and 0 none)
        -------
        penalization : the penalization send to the agent
        """
        self._numberOfMeasure += action

        return -error - self._numberOfMeasure * self._errorStep
        
