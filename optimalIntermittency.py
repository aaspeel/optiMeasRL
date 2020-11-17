""" 
The environment simulates the possibility of buying or selling a good. The agent can either have one unit or zero unit of that good. At each transaction with the market, the agent obtains a reward equivalent to the price of the good when selling it and the opposite when buying. In addition, a penalty of 0.5 (negative reward) is added for each transaction.
Two actions are possible for the agent:
- Action 0 corresponds to selling if the agent possesses one unit or idle if the agent possesses zero unit.
- Action 1 corresponds to buying if the agent possesses zero unit or idle if the agent already possesses one unit.
The state of the agent is made up of an history of two punctual observations:
- The price signal
- Either the agent possesses the good or not (1 or 0)
The price signal is build following the same rules for the training and the validation environment. That allows the agent to learn a strategy that exploits this successfully.
"""

import numpy as np
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt

from deer.base_classes import Environment

from keras.models import Sequential

from utils.sequences_treatment import *

# memories
sigmaMEMORY=5
observationsMEMORY=5


DEBUGMODE=False

class OptimalIntermittency(Environment):
    
    def __init__(self, estimator, rewarder, objectives, observations,rng):
        """ Initialize environment.

        Parameters
        -----------
        estimator : a RNN or a Kalman Filter that tries to estimate objectives from observations
        rewared : a class inheriting Rewared to compute the reward of the agent
        rng : the numpy random number generator
        """
        
        if DEBUGMODE:
            print('DEBUG: we are in __init__')
            
        # Set estimator
        self._estimator=estimator
        
        # Set rewarder
        self._rewarder=rewarder

        self._outOfRangeValue=estimator.outOfRangeValue()
        self._sigmaMEMORY=sigmaMEMORY
        self._observationsMEMORY=observationsMEMORY
        
        self._random_state=rng
        
        (self._numberSamples_all,self._T,self._n_dim_obj)=np.shape(objectives)
        (_,_,self._n_dim_obs)=np.shape(observations)
        
        self._objectives_all=objectives
        self._observations_all=observations
        
        # data for training
        self._objectives_train=objectives[:self._numberSamples_all//2,:,:]
        self._observations_train=observations[:self._numberSamples_all//2,:,:]
        (self._numberSamples_train,_,_)=np.shape(self._objectives_train)
        self._remain_samples_train=[] # will be initialized in reset()
        
        # data for validation 
        self._objectives_valid=objectives[self._numberSamples_all//2:,:,:]
        self._observations_valid=observations[self._numberSamples_all//2:,:,:]
        (self._numberSamples_valid,_,_)=np.shape(self._objectives_valid)
        self._remain_samples_valid=[] # will be initialized in reset()
        
        self._counter_action=0
        self._counter_measurement=0
        
        self._last_ponctual_observation = [0,self._n_dim_obs*[self._outOfRangeValue]] # At each time step, the observation is made up of two elements, each scalar
        
        print('Environment parameters')
        print('  REWARDER=',str(self._rewarder))
        print('  sigmaMEMORY=',sigmaMEMORY)
        print('  observationsMEMORY=',observationsMEMORY)
        print('Sequences parameters')
        print('  outOfRangeValue=',self._outOfRangeValue)
        print('  numerSamples=',self._numberSamples_all)
        print('  T=',self._T)
        print('  n_dim_obj=',self._n_dim_obj)
        print('  n_dim_obs=',self._n_dim_obs)
        
    def reset(self, mode):
        """ Resets the environment for a new episode.

        Parameters
        -----------
        mode : int
            -1 is for the training phase, others are for validation/test.

        Returns
        -------
        list
            Initialization of the sequence of observations used for the pseudo-state; dimension must match self.inputDimensions().
            If only the current observation is used as a (pseudo-)state, then this list is equal to self._last_ponctual_observation.
        """
        if DEBUGMODE:
            print('DEBUG: we are in reset')
        
        self._counter_action=0
        self._counter_measurement=0
        
        # set the correct data
        if mode==-1: # training mode
            self._objectives=self._objectives_train
            self._observations=self._observations_train
            self._numberSamples=self._numberSamples_train
            self._remain_samples=self._remain_samples_train
            # Modifying self._remain_samples also modify self._remain_samples_train
        else: # validation mode
            self._objectives=self._objectives_valid
            self._observations=self._observations_valid
            self._numberSamples=self._numberSamples_valid
            self._remain_samples=self._remain_samples_valid
            # Modifying self._remain_samples also modify self._remain_samples_valid
        
        # Select a random sample. Ensure all samples have been used before re-using one.
        # If no remaining samples, generates a new complete list
        if len(self._remain_samples)==0:
            self._remain_samples[:]=list(range(self._numberSamples))
            # 'list[:]=' instead of 'list=' conserves the same pointer, i.e. modify also self._remain_samples_currentMode
        
        # Select randomly a remaining sample and remove it from the list of remaining samples.
        self._currentSample=self._remain_samples.pop(np.random.randint(len(self._remain_samples)))
        
        # reset the estimator
        self._estimator.reset()
        
        # reset the rewarder
        self._rewarder.reset()
        
        self._mode=mode # for debug
        
        return [sigmaMEMORY*[0],observationsMEMORY*[self._n_dim_obs*[self._outOfRangeValue]]]


    def act(self, action):
        """ Performs one time-step within the environment and updates the current observation self._last_ponctual_observation

        Parameters
        -----------
        action : int
            Integer in [0, ..., N_A] where N_A is the number of actions given by self.nActions()

        Returns
        -------
        reward: float
        """
        if DEBUGMODE:
            print('DEBUG: we are in act')
        
        # current data
        current_objectives=self._objectives[self._currentSample,self._counter_action,:]
        current_observations=self._observations[self._currentSample,self._counter_action,:]
        current_objectives=current_objectives.reshape(1,1,-1)
        current_observations=current_observations.reshape(1,1,-1)
        
        sigma=np.array(action)
        
        # corrupt current_observations with mask format
        current_observations_mask = corruptSequence_mask(current_observations,sigma)
        
        # corrupt current_observations with outOfRange format
        current_observations_outOfRange = current_observations_mask.filled(self._outOfRangeValue)
        
        # estimate
        current_objectives_est = self._estimator.estimate(current_observations_mask) # shape (1,1,1)
            
        # compute estimation error
        error=(current_objectives-current_objectives_est).reshape((-1)) # to form an error vector
        
        # compute reward
        reward = self._rewarder.get(error, action)
        
        current_observations = np.reshape(current_observations_outOfRange,(-1))
        
        self._last_ponctual_observation[0] = action
        self._last_ponctual_observation[1] = current_observations_outOfRange
        
        self._counter_action+=1
        self._counter_measurement+=action
        
        return reward
    

    def summarizePerformance(self, test_data_set, *args, **kwargs):
        """
        This function is called at every PERIOD_BTW_SUMMARY_PERFS.
        Parameters
        -----------
            test_data_set
        """
        if DEBUGMODE:
            print('DEBUG: we are in summarizePerformance')
            
        #print('---------------- Summary Perf ----------------')
        observations=test_data_set.observations()
        sigma=observations[0]
        observations_c=observations[1]
        print("np.shape(sigma):",np.shape(sigma))
        print("np.shape(obserbations_c:",np.shape(observations_c))
        #print('mode:',self._mode)
        #print('currentSample:',self._currentSample)
        #print('remain_samples:',self._remain_samples)
        #print('counter_action:',self._counter_action)
        #print('A weight of the RNN:',self._estimator.get_layer(index=0).get_weights()[0][0][0])
        print('Summary Perf. Num measurements:',self._counter_measurement,'- Num actions:',self._counter_action,'- Mean sigma:',self._counter_measurement/self._counter_action)
        #print()
        #print('sigma',np.shape(sigma),':\n',sigma)
        #print('yc',np.shape(yc),':\n',yc)
        #print('----------------------------------------------')
        #print()


    def inputDimensions(self):
        if DEBUGMODE:
            print('DEBUG: we are in inputDimensions')
        return [(sigmaMEMORY,),(observationsMEMORY,self._n_dim_obs)]     # We consider an observation made up of an history of 
                                # - the last sigmaMEMORY for the first scalar element obtained
                                # - the last observationsMEMORY for the second scalar element

    def nActions(self):
        if DEBUGMODE:
            print('DEBUG: we are in nActions')
        return 2                # The environment allows two different actions to be taken at each time step


    def inTerminalState(self):
        if DEBUGMODE:
            print('DEBUG: we are in Terminal State.')
        return (self._counter_action==self._T)


    def observe(self):
        if DEBUGMODE:
            print('DEBUG: we are in observe')
        return np.array(self._last_ponctual_observation)


def main():
    # Can be used for debug purposes
    rng = np.random.RandomState(123456)
    myenv = MyEnv(rng)

    print (myenv.observe())
    
if __name__ == "__main__":
    main()
