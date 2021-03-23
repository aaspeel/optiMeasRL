""" 
Class kalmanEstimator
"""

from estimators.estimator import Estimator
from utils.particleFilter import *

class ParticleFilterEstimator(Estimator):
    
    def __init__(self,seeAction=True,seeMeasurement=True,seeEstimate=False,seeTime=False):
        """
        Construct the estimator.
        """
        self._pf=loadPF()
        self._pf.init_filter() #to create the particules
        
        #TOCHANGE
        self._n_dim_state=self._pf.particles.shape[1]
        self._n_dim_meas=self._pf.obs_dim# change of notation
        self._n_dim_obj=self._pf.output_dim
        #/TOCHANGE

        self._seeAction=seeAction
        self._seeMeasurement=seeMeasurement
        self._seeEstimate=seeEstimate
        self._seeTime=seeTime
        
        self.reset()
    
    
    def reset(self):
        """
        Initialize the constructor if necessary.
        """
        self._pf.init_filter()
        
        # reset observations
        self._last_action=0

        #TOCHANGE
        self._last_measurement_outOfRange=self._n_dim_meas*[self.outOfRangeValue()]
        self._last_estimate=self._n_dim_obj*[self.outOfRangeValue()] # could be different
        
        self._time=-1
    
    
    #Estimation for one serie
    def estimate(self,measurement_corrupted):
        """
        Return the estimate from corrupted informations.
        """
        # compute the estimate and update the internal state for Kalman filtering
        current_objective_est = PFFilterOne(self._pf, measurement_corrupted.reshape(-1))

        
        
        # storage for observation
        if measurement_corrupted.mask[0]: # masked
            self._last_action=0
        else:
            self._last_action=1

        self._last_measurement_outOfRange=measurement_corrupted.filled(self.outOfRangeValue())
        self._last_estimate=current_objective_est
        self._time+=1
        
        return current_objective_est
    
    
    def observe(self):
        """
        Return an observation to help the reinforcement learning agent.
        """
        observation=[]
        if self._seeAction:
            observation.append( self._last_action )
        if self._seeMeasurement:
            observation.append( self._last_measurement_outOfRange )
        if self._seeEstimate:
            observation.append( self._last_estimate )
        if self._seeTime:
            observation.append( 1-1/(self._time+2) ) # to represent the current time in [0,1[
        
        return observation
    
    
    def observationsDimensions(self):
        """
        Facultative
        Return the shape of an obsevation (including the action and the history size).
        """
        sigmaHistorySize=5
        measurementHistorySize=5
        estimateHistorySize=5
        
        dim=[]
        if self._seeAction:
            dim.append( (sigmaHistorySize,) )
        if self._seeMeasurement:
            pass
            #dim.append( (measurementHistorySize,self._n_dim_meas) )
        if self._seeEstimate:
            pass
            #dim.append( (estimateHistorySize,self._n_dim_obj) )
        if self._seeTime:
            dim.append( (1,) )
        
        return dim
    
    
    def estimateAll(self,measurements_corrupted):
        """
        Return the estimate from corrupted informations.
        """
        estimated_objectives = PFFilterAll(self._pf,measurements_corrupted)

        return estimated_objectives
    
    #TOCHANGE
    def generateSequence(self, T, numberSamples=1):
        """
        Facultative, generate sequences for which the estimator is designed.
        """
        (objectives,measurements,_)=samplePFSequence(self._pf,T,numberSamples=numberSamples)
        
        return (objectives,measurements)
    
    #TOCHANGE
    def summarize(self):
        """
        Facultative
        Print a summary of the predictor.
        """
        print('Particle filter estimator')
        print('  observationsDimensions:',self.observationsDimensions())
        print('  seeAction=',self._seeAction)
        print('  seeMeasurement=',self._seeMeasurement)
        print('  seeEstimate=',self._seeEstimate)
        print('  seeTime=',self._seeTime)
        
    