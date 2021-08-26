""" 
Class particleEstimator
"""

from estimators.estimator import Estimator
from utils.particleFilter import *

class ParticleFilterEstimator(Estimator):
    
    def __init__(self, T, windowSize, threshold, generatorType, seeAction=True,seeMeasurement=True,
                 seeEstimate=False,seeTime=False, seeSumAction=False):
        """
        Construct the estimator.
        """
        if generatorType == "spring":
            self._pf = loadPF_spring(T)
        elif generatorType == "benchmark":
            self._pf = loadPF_benchmark(T)
        else:
            self._pf = loadPF(T)
            
        self._pf.generatorType=generatorType
        
        print(type(self._pf))
        self._pf.init_filter() #to create the particules
        
        #TOCHANGE
        self._n_dim_state=self._pf.particles.shape[1]
        self._n_dim_meas=self._pf.obs_dim# change of notation
        self._n_dim_obj=self._pf.output_dim
        
        
        self._seeAction=seeAction
        self._seeMeasurement=seeMeasurement
        self._seeEstimate=seeEstimate
        self._seeTime=seeTime
        self._seeSumAction = seeSumAction
        
        self._windowSize = windowSize
        self._threshold = threshold
        self._T = T
        
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
        self._action_history = []
        self._sumAction = 0
    
    #Estimation for one time step
    def estimate(self,measurement_corrupted):
        """
        Return the estimate from possibly corrupted information.
        """
        current_objective_est = PFFilterOne(self._pf, measurement_corrupted.reshape(-1), self._time+1)

        # storage for observation
        if measurement_corrupted.mask[0]: # masked
            self._last_action=0
        else:
            self._last_action=1

        self._last_measurement_outOfRange=measurement_corrupted.filled(self.outOfRangeValue())
        self._last_estimate=current_objective_est
        self._time+=1
        
        if len(self._action_history) < self._windowSize:
            self._action_history.append(self._last_action)
        else:
            del(self._action_history[0])
            self._action_history.append(self._last_action)
        
        self._sumAction = sum(self._action_history)/self._threshold
        
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
            observation.append( self._time/self._T ) # to represent the current time in [0,1[
        if self._seeSumAction:
            observation.append( self._sumAction)
        
        return observation
    
    
    def observationsDimensions(self):
        """
        Return the shape of an obsevation (including the action and the history size).
        Used by the environment
        """
        sigmaHistorySize=12 # T-1
        measurementHistorySize=12 # T-1
        estimateHistorySize=12 # T-1
        
        dim=[]
        if self._seeAction:
            dim.append( (sigmaHistorySize,) )
        if self._seeMeasurement:
            dim.append( (measurementHistorySize,self._n_dim_meas) )
        if self._seeEstimate:
            dim.append( (estimateHistorySize,self._n_dim_obj) )
        if self._seeTime:
            dim.append( (1,) )
        if self._seeSumAction:
            dim.append( (1,) )
        
        return dim
    
    
    def estimateAll(self,measurements_corrupted):
        """
        Return the estimate from corrupted informations.
        """
        estimated_objectives = PFFilterAll(self._pf,measurements_corrupted)

        return estimated_objectives
    

    def generateSequence(self, T, numberSamples=1):
        """
        Facultative, generate sequences for which the estimator is designed.
        """
        (objectives,measurements,_)=samplePFSequence(self._pf,T,numberSamples=numberSamples)
        
        return (objectives,measurements)


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
        print('  seeSumAction=', self._seeSumAction)
        
    