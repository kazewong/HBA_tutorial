import numpy as np
from scipy.stats import beta
from scipy.special import erf
from scipy.interpolate import interp1d

def inverse_transform_sampling(bins,pdf,nSamples = 10000):
  cumValue = np.zeros(bins.shape)
  cumValue[1:] = np.cumsum(pdf[1:]*np.diff(bins))
  cumValue /= cumValue.max()
  inv_cdf = interp1d(cumValue,bins)
  r = np.random.rand(nSamples)
  return inv_cdf(r)

def M1_distribution(m1,index,mmin=5,mmax=50):
  try:
    output = m1.copy()
    index_in = np.where((m1>=mmin)*(m1<=mmax))[0]
    index_out = np.where((m1<mmin)+(m1>mmax))[0]
    normalization = ((mmax)**(1+index)-mmin**(1+index))/(1+index)
    output[index_out] = 1e-30
    output[index_in] = m1[index_in]**index/normalization
  except ZeroDivisionError:
    output = m1.copy()
    index_in = np.where((m1>=mmin)*(m1<=mmax))[0]
    index_out = np.where((m1<mmin)+(m1>mmax))[0]
    normalization = np.log(mmax)-np.log(mmin)
    output[index_out] = 1e-30
    output[index_in] = m1[index_in]**index/normalization
  return output

def spinAngle_distribution(cost1,cost2,sigma1,sigma2,xi):
	g1 = np.exp(-(1-cost1)**2/(2*sigma1**2))/(sigma1*erf(np.sqrt(2)/sigma1))
	g2 = np.exp(-(1-cost2)**2/(2*sigma2**2))/(sigma2*erf(np.sqrt(2)/sigma2))
	return (1.-xi)/4+2*xi/np.pi*g1*g2

def EVtoab(E,V):
	a = (E**2-E**3-E*V)/V
	b = (E-1)*(V-E+E**2)/V
	return a,b

def sampling_angle(sigma1,sigma2,xi,Nsample=10000):
	out1 = np.array([])
	out2 = np.array([])
	while out1.size < Nsample:
		cost1 = np.random.uniform(-1,1,Nsample)
		cost2 = np.random.uniform(-1,1,Nsample)
		randomN = np.random.uniform(0,spinAngle_distribution(1.,1.,sigma1,sigma2,xi),Nsample)
		pcos = spinAngle_distribution(cost1,cost2,sigma1,sigma2,xi)
		index = np.where(pcos>randomN)[0]
		out1 = np.append(out1,cost1[index])
		out2 = np.append(out2,cost2[index])
	return out1[:Nsample],out2[:Nsample]


