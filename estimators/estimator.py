""" 
Class estimator
"""

class Estimator:
    
    def __init__(self):
        """
        Construct the estimator.
        """
        self.reset()
        
    def reset(self):
        """
        Reset the estimator if necessary (initial conditions,...).
        """
        pass
    
    def estimate(self,observation_corrupted):
        """
        Return the estimate from corrupted informations.
        observation_corrupted is a numpy masked array.
        Must work (at least) for shape(observation_corrupted)=(1,1,n_dim_observations).
        """
        return NotImplementedError()
    
    
    def outOfRangeValue(self):
        """
        Return a value out of the range of the sequence of measurements.
        """
        return -1
    
    def extraInfo(self):
        """
        Facultative
        Return extra information to help the reinforcement learning agent.
        """
        return None
    
    def shapeExtraInfo(self):
        """
        Facultative
        Return the shape of the extra information.
        """
        return ()
    
    def summarize(self):
        """
        Facultative
        Print a summary of the predictor.
        """
        print("No function summarize() implemented in default Estimator class.")
        
    def estimateAll(self,observations_corrupted):
        """
        Facultative
        Estimate for all the observations_corrupted
        Contrary to the estimate function, observations_corrupted can have size (numberSamples,T,n_dim_obs)
        """
        return NotImplementedError()
        
    def generateSequence(self,T,numberSamples=1):
        """
        Facultative
        Generate sequences ( for which the estimator is designed.
        Return (objectives,observations) with shapes (numberSamples,T,:)
        """
        return NotImplementedError()
    