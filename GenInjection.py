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
parser.add_argument('--OutputDir',type=str,help='Output directory',required=True)
parser.add_argument('--Tag',type=str,help='Tag of files',default='Injection',required=False)

args = parser.parse_args()
OutDir = args.OutputDir
Tag = args.Tag
Nsample = args.Nsample

Maxis = np.linspace(3,100,10000)
qaxis = np.linspace(0,1,10000)
costaxis = np.linspace(-1,1,10000)

def GenCatalog(alphaM,betaM,sigma1,sigma2,Mmin=5,Mmax=50,spinME=0.5,spinMV=0.05,xi=0.5,Nsample=10000):
	m1 = inverse_transform_sampling(Maxis,M1_distribution(Maxis,alphaM,mmin=Mmin,mmax=Mmax),Nsample)
	q = inverse_transform_sampling(qaxis,M1_distribution(qaxis,betaM,mmin=0,mmax=1),Nsample)
	spinA,spinB = EVtoab(spinME,spinMV) 
	spinMag1 = beta(spinA,spinB).rvs(Nsample)
	spinMag2 = beta(spinA,spinB).rvs(Nsample)
	spinAng1,spinAng2 = sampling_angle(sigma1,sigma2,xi,Nsample)
	return np.array([m1,q,spinMag1,spinMag2,spinAng1,spinAng2])


design = [-2,2.,0.5,0.7]

output = GenCatalog(*design,Nsample=Nsample)

np.savez(OutDir+Tag,posterior_vector=output.T,prior_vector=np.ones(output.T.shape[0]),length_array=np.arange(output.T.shape[0]),Tobs=1)
