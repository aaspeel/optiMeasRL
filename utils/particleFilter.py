from utils.pfilter import *
import numpy as np
import copy
from math import sin

def loadPF(T=50):
    """ Create a particle filter with the tumour motion model
    """
    
    resample = None #systematic_resample
    dyn_noise = 1
    obs_noise = 1
    w_sigma = 1
    output_dim = 1
    obs_dim= 1
    t_step = 0.25
       
    noise_f = lambda x, **kwargs: x + np.random.normal(0, dyn_noise, x.shape)
    weight = lambda x,y, **kwargs : squared_error(x, y, sigma=w_sigma)
    
    def obs_noise_fn(x, **kwargs):
        
        return (x + np.random.normal(0, obs_noise, obs_dim)).reshape((x.shape[0],obs_dim))
        
        
    def rand_sin_prior(n):
        x = np.zeros((n,3))
        
        x[:,0] = 8.8 + 15.5 * np.random.uniform(0,1,n)
        x[:,1] = 1.3 + 2 * np.random.uniform(0,1,n)
        x[:,2] = - 5.8 + 12 * np.random.uniform(0,1,n)
        
        return x
        
    def rand_sin_dyn(x, **kwargs):
        """
        x: 4D array [a,b,omega, t]
        """
        #Apply Gaussian noise for a and b with variance= 1
        x[:,0] += np.random.normal(0, 1)
        x[:,1] += np.random.normal(0, 1)
        
        
        #clip values:
        x[:,0] = np.clip(x[:,0],8.8, 24) #8.8mm <= a <= 24mm
        x[:,1] = np.clip(x[:,1], -5.8, 5.8) #-5.8mm <= b <= 5.8mm
        
        
        return x
    
    
    def rand_sin_obs(x, t): 
        return x[:,0] * np.sin(t * x[:,2] * t_step) + x[:,1]
    
        
    def rand_sin_transf(x, weights, t):
        return (x[:,0] * np.sin(t * x[:,2] * t_step) + x[:,1]).reshape((x.shape[0],output_dim))
    

    pf = ParticleFilter(
        prior_fn = rand_sin_prior,
        observe_fn=rand_sin_obs,
        resample_fn=resample,
        n_particles=1000,
        dynamics_fn=rand_sin_dyn,
        noise_fn=noise_f,
        weight_fn=weight,
        resample_proportion=0.01, #0.02
        column_names=None,
        internal_weight_fn=None,
        transform_fn=rand_sin_transf,
        n_eff_threshold=1.0,
        obs_noise_fn = obs_noise_fn,
        output_dim = output_dim,
        obs_dim = obs_dim,
        dyn_noise = dyn_noise,
        T=T,
    )
    
    return pf


def samplePFSequence(pf,T,numberSamples=2):
    """ 
    Return an array with [z,obs,part] where
    - z: objective
    - obs (y): noisy observation
    - part (x): state of the model
    """

    #Pre allocation
    z = np.zeros((numberSamples,T,pf.output_dim))
    obs = np.zeros((numberSamples,T,pf.obs_dim))
    particles = np.zeros((numberSamples,T,pf.particles.shape[1]))

    
    #Create a copy of pf and change the number of particles to match numberSample
    #because we want each particle to give a trajectory
    pf2 = copy.deepcopy(pf)
    pf2.n_particles =numberSamples
    pf2.n_eff_threshold = 0
    pf2.resample_proportion = 0
    pf2.dyn_noise = 0.1
    
    pf2.init_filter()
    #update the weights to be of size numberSamples
    pf2.weights = np.ones(pf2.particles.shape[0]) / pf2.n_particles

    #Create time array:
    ts = [{"t":t} for t in np.linspace(0, T, T)]   
    
    for i in range(0,T):
        pf2.update(**ts[i])
        z[:,i,:] = pf2.transformed_particles
        obs[:,i,:] = pf2.obs_noise_fn(pf2.hypotheses)
        particles[:,i,:] = pf2.particles
    
    return z, obs, particles

    
def corruptPFSequence(observations,sigma):
    return


def PFFilterAll(pf,observations):
    (numberSamples,T,_)=np.shape(observations)

    result = np.zeros((numberSamples,T, pf.output_dim))
    
    for i in range(numberSamples):
        result[i,:,:] = PFFilterOne(pf, observations[i])
        
    return result


#Return the mean estimation given the observations
def PFFilterOne(pf, observation):
    
    #corrupt the observation given its mask and the specific value np.nan
    obs = copy.deepcopy(observation) #deepcopy to not corrupt the given list
    np.place(obs, obs.mask == True, np.nan)

    #Create the time array
    ts = [{"t":t} for t in np.linspace(0, pf.T, pf.T)]
    
    states = apply_filter(pf, obs, inputs=ts)
    
    transformed_particles = states["particles"] #z = h(x)
    weights = states["weights"]

    
    means = np.zeros((transformed_particles.shape[0],transformed_particles.shape[2]))
    #particles [T, n_particles, dim_particle]
    for i in range(means.shape[1]):
        means[:,i] = np.sum(transformed_particles[:,:,i] * weights, axis=1)

    return means



    
def apply_filter(pf, ys, inputs=None):
    """Apply filter pf to a series of observations (time_steps, h)  and return a dictionary:    
        particles: an array of particles (time_steps, n, d)
        weights: an array of weights (time_steps,)        
    """

    states = []
    pf.init_filter()  # reset
    for i,y in enumerate(ys):
        if inputs is None:
            pf.update(y)
        else:
            pf.update(y, **inputs[i])
            
        states.append([pf.transformed_particles, np.array(pf.original_weights)])
    return {
        name: np.array([s[i] for s in states])
        for i, name in enumerate(["particles", "weights"])
    }