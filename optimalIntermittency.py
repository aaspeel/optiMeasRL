import numpy as np
from deer.base_classes import Environment
from utils.sequences_treatment import *

DEBUGMODE=False

class OptimalIntermittency(Environment):
    
    def __init__(self, estimator, rewarder, objectives_train, measurements_train, objectives_valid, measurements_valid, rng):
        """ Initialize environment.

        Parameters
        -----------
        estimator : a RNN or a Kalman Filter that tries to estimate objectives from measurements
        rewarder : a class inheriting Rewarder to compute the reward of the agent
        rng : the numpy random number generator
        """
        
        if DEBUGMODE:
            print('DEBUG: we are in __init__')
            
        # Set estimator
        estimator.reset()
        self._estimator=estimator
        
        # Set rewarder
        rewarder.reset()
        self._rewarder=rewarder
        
        self._outOfRangeValue=estimator.outOfRangeValue()
        self._random_state=rng

        (_,_,self._n_dim_obj)=np.shape(objectives_train)
        (_,_,self._n_dim_meas)=np.shape(measurements_train)
        
        # data for training
        self._objectives_train=objectives_train
        self._measurements_train=measurements_train
        (self._numberSamples_train,_,_)=np.shape(self._objectives_train)
        self._remain_samples_train=[] # will be initialized in reset()
        
        # data for validation 
        self._objectives_valid=objectives_valid
        self._measurements_valid=measurements_valid
        (self._numberSamples_valid,_,_)=np.shape(self._objectives_valid)
        self._remain_samples_valid=[] # will be initialized in reset()
        
        self._current_time=0
        self._counter_measurement=0
        
        self._last_ponctual_observation = self._estimator.observe() # estimator has been reset.
        
        print('Environment parameters')
        print('  inputDimensions=',self.inputDimensions())
        print('Sequences parameters')
        print('  outOfRangeValue=',self._outOfRangeValue)
        print('  n_dim_obj=',self._n_dim_obj)
        print('  n_dim_meas=',self._n_dim_meas)
        print('  numberSamples_train',self._numberSamples_train)
        print('  numberSamples_valid',self._numberSamples_valid)
        self._estimator.summarize()
        self._rewarder.summarize()
        
        
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
        
        self._current_time=0
        self._counter_measurement=0
        
        self._mode=mode # to identify the inferenceMode
        
        # set the correct data
        if mode==-1: # training mode
            self._objectives=self._objectives_train
            self._measurements=self._measurements_train
            self._numberSamples=self._numberSamples_train
            self._remain_samples=self._remain_samples_train
            # Modifying self._remain_samples also modify self._remain_samples_train
            
        elif mode==0: # validation mode
            self._objectives=self._objectives_valid
            self._measurements=self._measurements_valid
            self._numberSamples=self._numberSamples_valid
            self._remain_samples=self._remain_samples_valid
            # Modifying self._remain_samples also modify self._remain_samples_valid
            
        elif mode>=1: #  mode==1: test/inference. mode==2: forced mode.
            # must be defined from outside using self.setTestData():
                    # - self._objectives_test
                    # - self._measurements_test
                    # - preallocation quantities
            # must be defined if mode==2:
                    # - self._forcedActions
            
            self._objectives=self._objectives_test
            self._measurements=self._measurements_test
            self._numberSamples=self._numberSamples_test
            self._remain_samples=self._remain_samples_test
            
        else:
            raise ValueError("ERROR in OptimalIntermittency: mode=",mode,"is not supported.")
        
        # Duration of the current sequence
        (_,self._T,_)=np.shape(self._measurements)
        
        # Select a random sample. Ensure all samples have been used before re-using one.
        # If no remaining samples, generates a new complete list
        if len(self._remain_samples)==0:
            self._remain_samples[:]=list(range(self._numberSamples))
            # 'list[:]=' instead of 'list=' conserves the same pointer, i.e. modify also self._remain_samples_currentMode
        
        if mode==1: # inference mode. No randomness
            self._currentSample=self._remain_samples.pop(0)
        else:
            # Select randomly a remaining sample and remove it from the list of remaining samples.
            self._currentSample=self._remain_samples.pop(np.random.randint(len(self._remain_samples)))
        
        # reset the rewarder
        self._rewarder.reset()
        
        # reset the estimator
        self._estimator.reset()
        
        # construct the initial pseudo state by copying the observation of the estimator
        obsDim = self._estimator.observationsDimensions()
        obs = self._estimator.observe() # estimator has been reset
        initialPseudoState=len(obsDim)*[0] # preallocation
        for i in range(len(obsDim)):
            initialPseudoState[i]=obsDim[i][0]*[obs[i]]
        
        return initialPseudoState

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
            print('self._mode=',self._mode)
            print('self._currentSample=',self._currentSample)
        
        # In mode 2, actions are overridden
        if self._mode==2:
            action=self._forcedActions[self._currentSample,self._current_time]
        
        # current data
        current_objective=self._objectives[self._currentSample,self._current_time,:]
        current_measurement=self._measurements[self._currentSample,self._current_time,:]
        current_objective=current_objective.reshape(1,1,-1)
        current_measurement=current_measurement.reshape(1,1,-1)
        
        sigma=np.array(action)
        
        # corrupt current_measurements with mask format
        current_measurement_mask = corruptSequence_mask(current_measurement,sigma)
        
        # estimate
        current_objective_est = self._estimator.estimate(current_measurement_mask) # shape (1,1,1)
        
        # observe (must be after the estimation)
        self._last_ponctual_observation = self._estimator.observe()
            
        # compute estimation error
        error=(current_objective-current_objective_est).reshape((-1)) # to form an error vector
        
        # compute reward
        reward = self._rewarder.get(error, action)
        
        # If test mode (mode 1) or in forced mode (mode 2), store data
        if self._mode>=1:
            self._testResults_sigmas[self._currentSample,self._current_time]=action
            self._testResults_rewards[self._currentSample,self._current_time]=reward
            self._testResults_estimates[self._currentSample,self._current_time,:]=current_objective_est
        
        self._current_time+=1
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
        #observations=test_data_set.observations()
        #sigma=observations[0]
        #observations_c=observations[1]
        #print("np.shape(sigma):",np.shape(sigma))
        #print("np.shape(obserbations_c):",np.shape(observations_c))
        #print('In env - summarize: mode:',self._mode)
        #print('currentSample:',self._currentSample)
        #print('remain_samples:',self._remain_samples)
        #print('_current_time:',self._current_time)
        #print('A weight of the RNN:',self._estimator.get_layer(index=0).get_weights()[0][0][0])
        #if self._current_time!=0:
        #    print('Summary Perf. Num measurements:',self._counter_measurement,'- Current time:',self._current_time,'- Mean sigma:',self._counter_measurement/self._current_time)
        #else:
        #    print('Current time=0')
        #print()
        #print('sigma',np.shape(sigma),':\n',sigma)
        #print('yc',np.shape(yc),':\n',yc)
        #print('----------------------------------------------')
        #print()


    def inputDimensions(self):
        if DEBUGMODE:
            print('DEBUG: we are in inputDimensions')
        return self._estimator.observationsDimensions()

    def nActions(self):
        if DEBUGMODE:
            print('DEBUG: we are in nActions')
        return 2                # The environment allows two different actions to be taken at each time step


    def inTerminalState(self):
        if DEBUGMODE:
            print('DEBUG: we are in Terminal State.')
        return (self._current_time>=self._T)


    def observe(self):
        if DEBUGMODE:
            print('DEBUG: we are in observe')
        return self._last_ponctual_observation
        
    def setTestData(self, objectives_test, measurements_test):
        """
        give data for testing (mode=1) and preallocate for storage.
        """
        # store data
        self._objectives_test=objectives_test
        self._measurements_test=measurements_test
        self._remain_samples_test=[] # will be initialized in reset()
        
        (self._numberSamples_test,T_test,_)=np.shape(self._objectives_test)
        
        # preallocation to store test results
        self._testResults_sigmas=np.zeros((self._numberSamples_test,T_test))
        self._testResults_rewards=np.zeros((self._numberSamples_test,T_test))
        self._testResults_estimates=np.zeros((self._numberSamples_test,T_test,self._n_dim_obj))
        # we could add observations
        
    def getTestResults(self):
        """
        return the data stored during testing (mode=1)
        """
        return (self._testResults_sigmas, self._testResults_rewards, self._testResults_estimates)
    
    def setForcedActions(self,forcedActions):
        """
        Actions to use in forced mode (mode 2).
        forcedActions must have shape (numberSamples_test,T_test).
        Function setTestData() must have been called previously.
        """
        (numberSamples,T)=np.shape(forcedActions)
        (_,T_test,_)=np.shape(self._objectives_test) # Will throw an error if test data not defined
        
        if numberSamples!=self._numberSamples_test or T!=T_test:
            raise ValueError("ERROR in OptimalIntermittency/setForcedActions: incompatible dimensions.")
        
        self._forcedActions=forcedActions
        
        # re-initialize the data storage
        self._remain_samples_test=[] # will be initialized in reset()
        self._testResults_sigmas=np.zeros((numberSamples,T)) # will be equal to forcedActions
        self._testResults_rewards=np.zeros((numberSamples,T))
        self._testResults_estimates=np.zeros((numberSamples,T,self._n_dim_obj))
        # we could add observations


def main():
    # Can be used for debug purposes
    rng = np.random.RandomState(123456)
    myenv = MyEnv(rng)

    print (myenv.observe())
    
if __name__ == "__main__":
    main()
