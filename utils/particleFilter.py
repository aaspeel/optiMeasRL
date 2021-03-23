from utils.pfilter import *
import numpy as np
import copy

def loadPF():
# specify parameters
    
    #Example taken from simple linear dynamics
    #https://github.com/johnhw/pfilter/blob/master/examples/timeseries.ipynb
    dt = 0.25
    noise_dyn = 0.125
    noise_sigma = 1.0

    # linear dynamics
    D = np.array([[1, dt, 0.5*dt**2],
                [0, 1, dt],
                [0, 0, 1]])

    O = np.array([[1, 0, 0]])


    #independent_sample takes a list of variable
    initialization = lambda n: np.random.normal(0,1,(n,3))
    #independent_sample([norm(loc=0.5, scale=0.2).rvs])
    
    obs = lambda x:  x @ O.T
    resample = systematic_resample
    particles = 200
    dynamics = lambda x:   x @ D.T
    noise_f = lambda x: x + np.random.normal(0, noise_dyn, x.shape)
    
    weight = lambda x,y : squared_error(x, y, sigma=noise_sigma)
    transform = None
    
    output_dim = 3
    obs_dim= 1
    
    obs_noise_fn = lambda x: x + np.random.normal(0, noise_sigma, obs_dim)
    
    
    pf = ParticleFilter(
        prior_fn = initialization,
        observe_fn=obs,
        resample_fn=None,
        n_particles=200,
        dynamics_fn=dynamics,
        noise_fn=noise_f,
        weight_fn=weight,
        resample_proportion=0, #0.02
        column_names=None,
        internal_weight_fn=None,
        transform_fn=transform,
        n_eff_threshold=1.0,
        obs_noise_fn = obs_noise_fn,
        output_dim = output_dim,
        obs_dim = obs_dim,
    )
    
    return pf


def samplePFSequence(pf,T,numberSamples=1):
    #return objective (z = h(x)), y (obs) and x
    #z = pf.transformed_particles
    #obs = pf.hypotheses.
    #x dim = pf.d
    
    #+ noise on obs needs same simga than the one used for weights (squared_error by default)
    #faire noise_obs(observe_fn) comme noise(dynamics_fn)
    #pf.init_filter()  # reset
    #pf.update()

    #Pre allocation
    z = np.zeros((numberSamples,T,pf.output_dim))
    obs = np.zeros((numberSamples,T,pf.obs_dim))
    particles = np.zeros((numberSamples,T,pf.particles.shape[1]))

    
    #Create a copy of pf and change number of particles to match numberSample
    #because we want each particle to give a trajectory
    pf2 = copy.deepcopy(pf)
    pf2.n_particles =numberSamples
    pf2.weights = np.ones(pf2.n_particles) / pf2.n_particles
    pf2.init_filter()
    
    #Manually fill T=0 because no update() yet
    particles[:,0,:] = pf2.particles
    obs[:,0,:] = pf2.obs_noise_fn(pf2.observe_fn(pf2.particles))
    if pf2.transform_fn:
            z[:,0,:] = pf2.transform_fn(pf2.particles, pf2.weights)
    else:
        z[:,0,:] = pf2.particles   
    
    
    for i in range(1,T):
        pf2.update()
        z[:,i,:] = pf2.transformed_particles
        obs[:,i,:] = pf2.obs_noise_fn(pf2.hypotheses)
        particles[:,i,:] = pf2.particles
    

    
    print(z[:numberSamples,:,:].shape)
    
    return z, obs, particles

    

def corruptPFSequence(observations,sigma):
    return


def PFFilterAll(pf,observations):
    
    #False running to get the shape of transformed_particles:
    #pf.update()
    #print(pf.transformed_particles.shape)
    (numberSamples,T,_)=np.shape(observations)
    result = np.zeros((numberSamples,T, pf.output_dim)) #TOCHANGE see PFFilterOne
    
    for i in range(numberSamples):
        result[i,:,:] = PFFilterOne(pf, observations[i])
        
    return result


#Return the mean estimation given the observations
def PFFilterOne(pf, observation):
    
    #corrupt the observation given its mask and the specific value np.nan
    obs = copy.deepcopy(observation) #deepcopy to not corrupt the given list
    np.place(obs, obs.mask == True, np.nan)
    #TOCHANGE
    #obs = measurement_corrupted.filled(self.outOfRangeValue())
    states = apply_filter(pf, obs)

    particles = states["particles"] #z = h(x)
    weights = states["weights"]
    
    print(particles.shape)
    
    means = np.zeros((particles.shape[0],particles.shape[2]))
    #TOCHANGE: here we only take the mean of the first argument (x)
    #particles [T, n_particles, dim_particle]
    for i in range(means.shape[1]):
        means[:,i] = np.sum(particles[:,:,i] * weights, axis=1)

    return means


#https://github.com/johnhw/pfilter/blob/master/examples/timeseries.ipynb
def apply_filter(pf, ys, inputs=None):
    """Apply filter pf to a series of observations (time_steps, h)
        Returns: ["particles":z=h(x), "weights":list of weights]
        
        if ys[i] is None then there is no update of the particles for step i
    """

    states = []
    pf.init_filter()  # reset
    for i,y in enumerate(ys):
        if inputs is None:
            pf.update(y)
        else:
            pf.update(y, **inputs[i])
            
        states.append([pf.transformed_particles, np.array(pf.weights)])

    return {
        name: np.array([s[i] for s in states])
        for i, name in enumerate(["particles", "weights"])
    }
    
    
    