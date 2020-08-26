import numpy as np
from scipy.stats import beta
from scipy.special import erf
from scipy.interpolate import interp1d
from PhenomFunction import *
from pyDOE import *
from scipy.stats import uniform
import argparse

parser = argparse.ArgumentParser(description='Create training catalog')
parser.add_argument('--Nsample',type=int,help='Number of sample',required=True)
parser.add_argument('--Nsim',type=int,help='Number of simulation',required=True)
parser.add_argument('--OutputDir',type=str,help='Output directory',required=True)
parser.add_argument('--Tag',type=str,help='Tag of files',default='Injection',required=False)

args = parser.parse_args()
OutDir = args.OutputDir
Tag = args.Tag
Nsample = args.Nsample
Nsim = args.Nsim

Maxis = np.linspace(3,100,10000)
qaxis = np.linspace(0,1,10000)
costaxis = np.linspace(-1,1,10000)

def GenCatalog(alphaM,betaM,sigma1,sigma2,xi=0.5,Mmin=5,Mmax=50,spinME=0.5,spinMV=0.05,Nsample=10000):
	m1 = inverse_transform_sampling(Maxis,M1_distribution(Maxis,alphaM,mmin=Mmin,mmax=Mmax),Nsample)
	q = inverse_transform_sampling(qaxis,M1_distribution(qaxis,betaM,mmin=0,mmax=1),Nsample)
	spinA,spinB = EVtoab(spinME,spinMV) 
	spinMag1 = beta(spinA,spinB).rvs(Nsample)
	spinMag2 = beta(spinA,spinB).rvs(Nsample)
	spinAng1,spinAng2 = sampling_angle(sigma1,sigma2,xi,Nsample)
	return np.array([m1,q,spinMag1,spinMag2,spinAng1,spinAng2])

loc = [-3,1,0,0]
scale = [3,2,2,2]


design = lhs(4,samples=Nsim)
design[:,0] = uniform(loc[0],scale[0]).ppf(design[:,0])
design[:,1] = uniform(loc[1],scale[1]).ppf(design[:,1])
design[:,2] = uniform(loc[2],scale[2]).ppf(design[:,2])
design[:,3] = uniform(loc[3],scale[3]).ppf(design[:,3])

output = []
delete_index = []
for i in range(Nsim):
	try:
		output.append(GenCatalog(*design[i],Nsample=Nsample))
	except ValueError:
		print('Invalid expectation value and variance pair of spin magnitude.')
		print(design[i])
		delete_index.append(i)

print('The output number of simulation is '+str(Nsim-len(delete_index)))
design = np.delete(design,delete_index,0)
output = np.array(output)
sample = output.swapaxes(1,2).reshape(output.shape[0]*output.shape[2],-1)
design = np.repeat(design,output.shape[2],axis=0)
np.savez(OutDir+Tag,sample=sample,hyperParameter=design)
