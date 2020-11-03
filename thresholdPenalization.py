"""
The threshold penalization only penalize the error of the predictor
until we reach a certain number of measurements (threshold) then
the error becomes -1-error given that the signal is between 0 and 1
such that taking a new measure is too costly for the agent.
"""

class ThresholdPenalization(Penalization):

    def __init__(self, threshold):
        """
        Implement if needed.
        """
        self.threshold = threshold
        self.numberOfMeasure = 0


    def get(self, error, action, *args, **kwargs):
        """ Return the penalization for the RL agent given a certain criteria
        Arguments
        ---------
        error : the error made by the predictor
        action : the action taken by the agent (1 means measure taken and 0 none)
        -------
        penalization : the penalization send to the agent
        """

        self.numberOfMeasure += action
        if self.numberOfMeasure >= self.threshold:
            return -1-error

        return -error
        
