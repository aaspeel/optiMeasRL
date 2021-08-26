from utils.pfilter import *
import numpy as np
import copy
from math import sin


def loadPF(T=50):
    """ Create a particle filter with a particular model
    """
    
    return loadPF_tumor(T=T)


def loadPF_spring(T=50):
    
    #kf data
    initial_state_mean = np.array([0, 1])
    n_state = len(initial_state_mean)
    initial_state_covariance = np.array([[1,0],[0,1]])
    delta=0.5
    transition_matrix = np.array([[np.cos(delta), np.sin(delta)], [-np.sin(delta), np.cos(delta)]])
    observation_matrix = np.array([1,0])
    objective_matrix = np.array([1,0]) # Added by A. Aspeel
    transition_covariance = (1/(2*40**2))*np.array([[ delta-np.sin(delta)*np.cos(delta)  ,np.sin(delta)**2] ,
                                                           [np.sin(delta)**2 ,  delta+np.sin(delta)*np.cos(delta) ]])
    observation_covariance = np.array([1]) #[[1,0],[0,1]]
    
    #pf data
    w_sigma = 1 # for weighting
    output_dim = 1
    obs_dim = 1
    resample = None
    
    def prior_fn(n):
        x = np.random.multivariate_normal(initial_state_mean,
                                             initial_state_covariance, n)
        return x
    
    def dynamics_fn(x, t):
        res = np.dot(x, transition_matrix.T)
        return res
    
    def noise_fn(x, **kwargs):
        return x + np.random.multivariate_normal( np.zeros(n_state),
                                                 transition_covariance,
                                                 x.shape[0])
        return x
    
    def observe_fn(x, **kwargs):
        return np.dot(x, observation_matrix.T)
    
    def obs_noise_fn(x, **kwargs):
        if obs_dim < 2:
            #using normal because multivariate needs a cov matrix of min size (2,2)
            x += np.random.normal(0, observation_covariance, x.shape)                              
        else:
            x += np.random.multivariate_normal(np.zeros(n_state), observation_covariance,    
                                                 x.shape[0])
        return x.reshape((x.shape[0],obs_dim))
    
    def weight_linear(x,y, **kwargs): 
        return squared_error(x, y, sigma=w_sigma)
    
    def transform_fn(x, weights, **kwargs):
        return np.dot(x, objective_matrix.T).reshape((x.shape[0],output_dim))
    
    pf = ParticleFilter(
        prior_fn = prior_fn,
        observe_fn = observe_fn,
        resample_fn = resample,
        n_particles=100,
        dynamics_fn=dynamics_fn,
        noise_fn=noise_fn,
        weight_fn=weight_linear,
        resample_proportion=0.01,
        column_names=None,
        internal_weight_fn=None,
        transform_fn=transform_fn,
        n_eff_threshold=1.0,
        obs_noise_fn = obs_noise_fn,
        output_dim = output_dim,
        obs_dim = obs_dim,
        T=T,
    )
    
    return pf
    


def loadPF_benchmark(T=50):
    """ Create a particle filter with the tumour motion model
    """
    resample = None #systematic_resample
    dyn_noise = 1
    w_sigma = dyn_noise # for weighting
    output_dim = 1
    obs_dim = 1
    t_step = 1
    
    def prior_fn(n):
        return np.random.normal(0,5,n).reshape((n,1))
    
    def dynamics_fn(x, t):
        """
        fn
        x: 4D array [a,b,omega, t]
        """
        #Apply Gaussian noise on the 'real state' and update the time
        x = x/2 + 25*x/(1+x**2)+8*np.cos(1.2*t)
        return x
    
    def noise_fn(x, **kwargs):
        return x + np.random.normal(0, dyn_noise, x.shape) # add noise (not on the time variable)
    
    def observe_fn(x, **kwargs): 
        return x**2/20
    
    def obs_noise_fn(x, t):
        std=np.sin(0.25*t)+2
        return (x + np.random.normal(0, std, x.shape))#.reshape((x.shape[0],obs_dim))
    
    def weight(x, y, t):
        std = np.sin(0.25*t)+2
        return squared_error(x, y, sigma=std)
        
    def transform_fn(x, weights, **kwargs):
        return x

    pf = ParticleFilter(
        prior_fn = prior_fn,
        observe_fn = observe_fn,
        resample_fn = resample,
        n_particles=100,
        dynamics_fn=dynamics_fn,
        noise_fn=noise_fn,
        weight_fn=weight,
        resample_proportion=0.01,
        column_names=None,
        internal_weight_fn=None,
        transform_fn=transform_fn,
        n_eff_threshold=1.0,
        obs_noise_fn = obs_noise_fn,
        output_dim = output_dim,
        obs_dim = obs_dim,
        dyn_noise = dyn_noise,
        T=T,
    )
    
    return pf

def loadPF_tumor(T=50):
    """ Create a particle filter with the tumour motion model
    """
    
    resample = None #systematic_resample
    dyn_noise = 1
    obs_noise = 1
    w_sigma = 1
    
    sigma_a = 1
    sigma_b = 1
    sigma_omega = 0.005
    output_dim = 1
    obs_dim= 1
    t_step = 0.25
       
    #noise_f = lambda x, **kwargs: x + np.random.normal(0, dyn_noise, x.shape)
    def noise_fn(x, **kwargs):
        #Apply Gaussian noise for a and b with variance= 1
        x[:,0] += np.random.normal(0, sigma_a)
        x[:,1] += np.random.normal(0, sigma_b)
        x[:,2] += np.random.normal(0, sigma_omega)
        
        #clip values:
        x[:,0] = np.clip(x[:,0],8.8, 24) #8.8mm <= a <= 24mm
        x[:,1] = np.clip(x[:,1], -5.8, 5.8) #-5.8mm <= b <= 5.8mm
        x[:,2] = np.clip(x[:,2], 1.3, 2.1) # 1.3rad/s <= omega <= 2.1rad/s
        
        return x
        
        
    weight = lambda x,y, **kwargs : squared_error(x, y, sigma=w_sigma)
    
    def obs_noise_fn(x, **kwargs):
        return (x + np.random.normal(0, obs_noise, obs_dim)).reshape((x.shape[0],obs_dim))
        
        
    def rand_sin_prior(n):
        x = np.zeros((n,3))
        
        x[:,0] = 8.8 + 15.2 * np.random.uniform(0,1,n)
        x[:,1] = - 5.8 + 11.6 * np.random.uniform(0,1,n)
        x[:,2] = 1.3 +  0.8 * np.random.uniform(0,1,n)
        
        return x
    
    
    def rand_sin_dyn(x, **kwargs):
        """
        x: 4D array [a,b,omega, t]
        """
        return x
    
    
    def rand_sin_obs(x, t): 
        return x[:,0] * np.sin(t * x[:,2] * t_step) + x[:,1]
    
        
    def rand_sin_transf(x, weights, t):
        return (x[:,0] * np.sin(t * x[:,2] * t_step) + x[:,1]).reshape((x.shape[0],output_dim))
    

    pf = ParticleFilter(
        prior_fn = rand_sin_prior,
        observe_fn=rand_sin_obs,
        resample_fn=resample,
        n_particles=100,
        dynamics_fn=rand_sin_dyn,
        noise_fn=noise_fn,
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
    
    def tumour_noise_fn(x, **kwargs):
        #Apply Gaussian noise for a and b with variance= 1
        sigma_a = 1
        sigma_b = 1
        sigma_omega = 0
        x[:,0] += np.random.normal(0, sigma_a)
        x[:,1] += np.random.normal(0, sigma_b)
        
        #clip values:
        x[:,0] = np.clip(x[:,0],8.8, 24) #8.8mm <= a <= 24mm
        x[:,1] = np.clip(x[:,1], -5.8, 5.8) #-5.8mm <= b <= 5.8mm
        x[:,2] = np.clip(x[:,2], 1.3, 2.1) # 1.3rad/s <= omega <= 2.1rad/s    
        
        return x
        
    #Special case for pf tumour where omega must be noiseless
    if hasattr(pf2,'generatorType'):
        if pf2.generatorType == "tumour":
            pf2.noise_fn = tumour_noise_fn
        
    pf2.n_particles = numberSamples
    pf2.n_eff_threshold = 0
    pf2.resample_proportion = 0
    #IMPORTANT: the smaller, the smoother the trajectories
    pf2.dyn_noise = 0.1
    
    pf2.init_filter()
    #update the weights to be of size numberSamples
    pf2.weights = np.ones(pf2.particles.shape[0]) / pf2.n_particles

    #Create time array:
    ts = [{"t":t} for t in np.linspace(0, T, T)]   
    
    for i in range(0,T):
        pf2.update(**ts[i])
        z[:,i,:] = pf2.transformed_particles
        obs[:,i,:] = pf2.obs_noise_fn(pf2.hypotheses,**ts[i])
        particles[:,i,:] = pf2.particles
    
    return z, obs, particles


def PFFilterAll(pf,observations):
    """ Estimate for n numberSamples the whole time serie
    Arguments:
        - pf: a particleFilter objet
        - observation: list of shape [n samples,n time steps, obs dim] containing the 
            observation of the whole time serie for all samples
    Return:
        - result: the estimation of the particle filter of shape [n sample,
            n time step, output_dim]
    """
    #TODO change
    (numberSamples,T,_)=np.shape(observations)

    result = np.zeros((numberSamples,T, pf.output_dim))
    #Create time serie
    ts = [{"t":t} for t in np.linspace(0, T-1, T)]
    for i in range(numberSamples):
        #init filter
        pf.init_filter()
        
        #corrupt observation i:
        obs = copy.deepcopy(observations[i]) #deepcopy to not corrupt the given list
        np.place(obs, obs.mask == True, np.nan) # replace the masked values by nans because it is the convention used by pfilter
        result[i,:,:] = _apply_filter(pf, obs, inputs=ts)
        
    return result


def PFFilterOne(pf, observation, time_step):
    """Estimate one time step. Observation is a masked array.
    Mean in apply filter for One and All
    Arguments:
        - pf: a particleFilter objet
        - observation: list of size one containing the observation of the
            current time step
    Return:
        - mean: estimation of the particle filter (weighted sum of particles)
    """
    
    #corrupt the observation given its mask and the specific value np.nan
    obs = copy.deepcopy(observation) #deepcopy to not corrupt the given list
    np.place(obs, obs.mask == True, np.nan)
    #Create the time array
    ts = [{"t":time_step}] #for t in np.linspace(0, pf.T, pf.T)]
    
    return _apply_filter(pf, obs, inputs=ts)[0]

def _apply_filter(pf, ys, inputs=None):
    """
    Apply the particle filter over mutliple time steps (depending on the size of ys). The filter is not initialized
    Missing measurements in ys must be nan.
    Apply filter pf to a series of observations (time_steps, h) and return a dictionary:
        particles: an array of particles (time_steps, n, d)
        weights: an array of weights (time_steps,)
    """

    states = []
    for i,y in enumerate(ys):
        if inputs is None:
            pf.update(y)
        else:
            pf.update(y, **inputs[i])
            
        states.append([pf.transformed_particles, np.array(pf.original_weights)])
        
    states = {
        name: np.array([s[i] for s in states])
        for i, name in enumerate(["particles", "weights"])
    }
    transformed_particles = states["particles"] #z = h(x)
    weights = states["weights"]    
    
    means = np.zeros((transformed_particles.shape[0],transformed_particles.shape[2]))
    #particles [T, n_particles, dim_particle]
    for i in range(means.shape[1]):
        means[:,i] = np.sum(transformed_particles[:,:,i] * weights, axis=1)

    return means
