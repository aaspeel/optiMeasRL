"""
The threshold rewarder only gives a cost of taking a measure
when we reach a certain number of measurements (threshold) then
the error becomes -cost-error such that taking a new measure 
is too costly for the agent.
"""

from rewarder import Rewarder

class ThresholdRewarder(Rewarder):

    def __init__(self, threshold, windowSize, cost=2):
        """
        Arguments
        ---------
        threshold : the maximum number of measures allowed (must be > 0)
        cost : the cost of taking a measure when numberOfMeasure >= threshold
        """
        self._threshold = threshold
        self._cost = cost
        self._numberOfMeasure = 0
        self._errors = []
        self._winSize = windowSize
        self._window = []


    def reset(self):
        """
        Rest internal parameters
        """
        self._numberOfMeasure = 0
        self._errors = []
        self._window = []


    def get(self, error, action, *args, **kwargs):
        """ Return the reward for the RL agent given a certain criteria
        Arguments
        ---------
        error : the error made by the predictor
        action : the action taken by the agent (1 means measure taken and 0 none)
        Returns
        -------
        reward : the reward send to the agent
        """

        self.update(action)

        if self._numberOfMeasure >= self._threshold:
            err = -self._cost - error
            self._errors.append(err)
            return err
        self._errors.append(-error)

        return -error
    

    def update(self, action):
        """ Update the window and the number of measures
        Argument
        ---------
        action : the action taken by the agent (1 means measure taken and 0 none)
        Returns
        -------
        """

        if (len(self._window) + 1) >= self._winSize:
            del(self._window[0])
        
        self._window.append(action)
        self._numberOfMeasure = sum(self._window)


    def info(self):
        print("threshold: " + str(self._threshold))
        print("cost: " + str(self._cost))
        print("number of measures: " + str(self._numberOfMeasure))
        print("error for measure: " + str(self.get(0,0)))
        print("error history: " + str(self._errors))
        
