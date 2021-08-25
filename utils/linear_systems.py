"""
To manipulate linear systems and Kalman filtering.
"""

import numpy as np
import pylab as pl
from pykalman import KalmanFilter

def loadKF():
# specify parameters
    random_state = np.random.RandomState(0)
    
    delta=0.5
    
    transition_matrix = [[np.cos(delta), np.sin(delta)], [-np.sin(delta), np.cos(delta)]]
    observation_matrix = [[1,0]]
    objective_matrix = [[1,0]] # Added by A. Aspeel
    
    transition_covariance = (1/(2*40**2))*np.matrix([[ delta-np.sin(delta)*np.cos(delta)  ,np.sin(delta)**2] ,
                                                           [np.sin(delta)**2 ,  delta+np.sin(delta)*np.cos(delta) ]])
    
    observation_covariance = [[1]] #[[1,0],[0,1]]
    
    initial_state_mean = [0, 1]
    initial_state_covariance = [[1,0],[0,1]]
    
    kf = KalmanFilter(
        transition_matrices=transition_matrix,
        observation_matrices=observation_matrix,
        transition_covariance=transition_covariance,
        observation_covariance=observation_covariance,
        initial_state_mean=initial_state_mean,
        initial_state_covariance=initial_state_covariance
    )
    
    # Added by A. Aspeel
    kf.objective_matrix=np.array(objective_matrix)
    kf.n_dim_obj=np.shape(objective_matrix)[0]
    
    return kf

def sampleKFSequence(kf,T,numberSamples=1):
# sample from KalmanFilter model
    states=np.zeros((numberSamples,T,kf.n_dim_state))
    observations=np.ma.zeros((numberSamples,T,kf.n_dim_obs))
    objectives=np.ma.zeros((numberSamples,T,kf.n_dim_obj))
    for i in range(numberSamples):
        states[i,:,:],observations[i,:,:] = kf.sample(T)
        objectives[i,:,:] = np.dot(states[i,:,:],kf.objective_matrix.T)
        
    return objectives,observations,states

def corruptKFSequence(observations,sigma):
    # transform sigma into a (numberSamples,T,m) array without changing sigma
    (numberSamples,T,n_dim_obs)=np.shape(observations)
    sigma2=sigma.reshape([numberSamples,T,1])
    sigma2=np.tile(sigma2,(1,1,n_dim_obs))
    observations_corrupted = np.ma.array(observations,mask=(1-sigma2))

    return observations_corrupted

def KFFilterAll(kf,observations):
    (numberSamples,T,_)=np.shape(observations)
    filtered_states=np.zeros((numberSamples,T,kf.n_dim_state))
    filtered_objectives=np.zeros((numberSamples,T,kf.n_dim_obj))
    
    state_covariances=np.zeros((numberSamples,T,kf.n_dim_state,kf.n_dim_state))
    objective_covariances=np.zeros((numberSamples,T,kf.n_dim_obj,kf.n_dim_obj))
    
    for i in range(numberSamples):
        filtered_states[i,:,:],state_covariances[i,:,:,:]=kf.filter(observations[i,:,:])
        filtered_objectives[i,:,:]=np.dot(filtered_states[i,:,:],kf.objective_matrix.T)
        objective_covariances[i,:,:,:]=np.transpose(
            np.tensordot(
                kf.objective_matrix, np.dot(
                    state_covariances[i,:,:,:], kf.objective_matrix.T
                ),axes=(1,1)
            ),(1,0,2)
        ) 
    
    return filtered_objectives, objective_covariances, filtered_states, state_covariances

def KFFilterOne(kf, observation, state_mean=None, state_mean_covariance=None):
    if state_mean is None:
        state_mean=kf.initial_state_mean
    if state_mean_covariance is None:
        state_mean_covariance=kf.initial_state_covariance
    
    filtered_state, filtered_state_covariance=kf.filter_update(
        state_mean,
        state_mean_covariance,
        observation
    )
    
    filtered_objective=np.dot(kf.objective_matrix,filtered_state)
    filtered_objective_covariance=np.dot(
        kf.objective_matrix, np.dot(
            filtered_state_covariance, kf.objective_matrix.T
        )
    )
    
    return filtered_state, filtered_state_covariance, filtered_objective, filtered_objective_covariance
