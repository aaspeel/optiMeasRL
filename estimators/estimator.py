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
    
    def estimate(self,measurement_corrupted):
        """
        Return the estimate from corrupted informations.
        measurement_corrupted is a numpy masked array.
        Must work (at least) for shape(measurement_corrupted)=(1,1,n_dim_measurement).
        """
        return NotImplementedError()
    
    
    def observe(self):
        """
        Return an observation to help the reinforcement learning agent.
        """
        return None
    
    
    def observationsDimensions(self):
        """
        Facultative
        Return the shape of an obsevation (including the action and the history size).
        """
        return ()
    
    
    def outOfRangeValue(self):
        """
        Return a value out of the range of the sequence of measurements.
        """
        return -1
    
    
    def summarize(self):
        """
        Facultative
        Print a summary of the predictor.
        """
        print("No function summarize() implemented in default Estimator class.")
    
    
    def estimateAll(self,measurements_corrupted):
        """
        Facultative
        Estimate for all the measurements_corrupted
        Contrary to the estimate function, measurements_corrupted can have size (numberSamples,T,n_dim_meas)
        """
        return NotImplementedError()
        
    def generateSequence(self,T,numberSamples=1):
        """
        Facultative
        Generate sequences ( for which the estimator is designed.
        Return (objectives,measurements) with shapes (numberSamples,T,:)
        """
        return NotImplementedError()
        
    