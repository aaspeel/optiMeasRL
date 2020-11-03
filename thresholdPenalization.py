"""
The threshold penalization only penalize the error of the predictor
until we reach a certain number of measurements (threshold) then
the error becomes -1-error given that the signal is between 0 and 1
such that taking a new measure is too costly for the agent.
"""

from penalization import Penalization

class ThresholdPenalization(Penalization):

    def __init__(self, threshold):
        """
        Implement if needed.
        """
        self._threshold = threshold
        self._numberOfMeasure = 0
        self._errors = []


    def reset(self):
        """
        Rest internal parameters
        """
        self._numberOfMeasure = 0
        self._errors = []


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
        if self._numberOfMeasure >= self._threshold:
            self._errors.append(-1-error)
            return -1-error
        self._errors.append(-error)
        return -error
    
    def info(self):
        print("threshold: " + str(self._threshold))
        print("number of measures: " + str(self._numberOfMeasure))
        print("error for measure: " + str(self.get(0,0)))
        print("error history: " + str(self._errors))
        
