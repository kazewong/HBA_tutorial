import numpy as np
from scipy.stats import beta
from scipy.special import erf
from scipy.interpolate import interp1d
from PhenomFunction import *

data = np.load('./test.npz')
m1,q,sm1,sm2,sa1,sa2 = data['posterior_vector'].T
priorRange = [[-3,0],[1,3],[0,2],[0,2]]

def LogLike(cube):
	pm1 = M1_distribution(m1,cube[0],mmin=5,mmax=50)
	pq = M1_distribution(q,cube[1],mmin=0,mmax=1.)
	pcos = spinAngle_distribution(sa1,sa2,cube[2],cube[3],0.5)
	output = np.log(pm1*pq*pcos)
	for i in range(ndim):	
		if (cube[i]<priorRange[i][0])+(cube[i]>priorRange[i][1]):
			return -np.inf
	return np.sum(output[np.isfinite(output)])


nwalkers,ndim = 30,4
p0 = np.array([np.random.uniform(priorRange[i][0],priorRange[i][1],nwalkers) for i in range(ndim)]).T

print('Start sampling log posterior.')

import emcee
sampler = emcee.EnsembleSampler(nwalkers,ndim,LogLike)#,pool=pool)
p0 = sampler.run_mcmc(p0,100) # Burn-in steps
sampler.reset()
p0 = sampler.run_mcmc(p0.coords,500)

np.savez('./result',samples=sampler.flatchain.T)
#m2,m1 = np.sort(np.array([m1,m2]),axis=0)
