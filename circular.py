import numpy as np
from scipy import stats
from utils import sample_discrete_rvs
import utils


################ General circular coupling class #################

class CircularCoupler():
	"""
	N : int
		Number of samples
	reps : int
		Number of replications
	params : dict
		Dictionary of parameters for sampling.
		Examples:
			- For ising model, the parameter theta, dimensionality
			- For discrete state space, the transition matrix K
	"""
	
	def __init__(
		self, 
		N, 
		reps,
		params=None
	):
		self.N = N
		self.reps = reps
		self.params = params
		
	def pi0(self):
		"""
		Samples self.reps replications from an initial distribution

		Returns: a numpy array of shape d x reps, where d is dimensionality
		"""

		raise NotImplementedError()

	def sample_next(self, seed, xprev, yprev=None):
		"""
		Samples the next state from the chain given xprev.
		Can either use an explicit seed or "yprev" to implement couplings. 

		Returns: a numpy array of shape d x reps, where d is dimensionality
		"""

		raise NotImplementedError()

	def forward(self):
		"""
		Returns self.xs, a (N, d, reps)-shaped array.
		"""
		
		# Initial sample
		self.xs = [self.pi0()]
		for j in range(self.N - 1):
			self.xs.append(
				self.sample_next(seed=j, xprev=self.xs[-1])
			)
		
		# Make a numpy array
		self.xs = np.array(self.xs)
		return self.xs
	
	def backward(self):
		"""
		Return self.ys, a (N, d, reps)-shaped array.
		"""
		
		self.ys = [self.xs[-1]]
		for j in range(self.N - 1):
			self.ys.append(
				self.sample_next(seed=j, xprev=self.ys[-1])
			)
		
		self.ys = np.array(self.ys)
		return self.ys

	def convergence_times(self):
		"""
		Returns the first time at which y = x for each replication.
		"""
		flags = np.all(self.ys == self.xs, axis=1) # N x reps
		output = np.argmax(flags, axis=0).astype(float)
		output[flags[-1] == 0] = np.nan # account for cases which didn't converge
		return output
	
	def compute_estimates(self, func=None):
		"""
		pi is the stationary distribution
		"""
		if func is None:
			func = lambda x: x
			func = np.vectorize(func)
		
		converged = np.all(self.ys[-1] == self.xs[-1], axis=0)
		hatthetas = func(self.ys[:, :, converged]).mean(axis=0)
		return hatthetas, theta
	
	def auxiliary_diagnostics(self, k=None, r=10): # r must divide N
		if k is None:
			k = self.N / 2
		if self.N % r != 0:
			raise ValueError(f"r {r} must divide N {self.N}")

		# This follows the notation from Neal 1999
		cis = []
		for i in range(0, r):
			s = i*self.N // r
			# reps-dimensional vector
			zis = self.pi0()
			ci = np.zeros(self.reps)
			for t in range(s+1, s+k+1):
				if zis != self.ys[t-1]:
					break
				zis = self.sample_next(
					seed=t, xprev=zis, yprev=self.ys[t-1],
				)
				converged_flags = np.all(zis == self.ys[t-1], axis=0)
				ci[~converged_flags] = ci[~converged_flags] + 1
			cis.append(ci)

		return np.array(cis)

################ Markov Chain with Discrete State Space #################

class DiscreteCircularCoupler(utils.DiscreteSampler, CircularCoupler):
	""" Circular coupler for discrete markov chains """

	def __init__(
		self, 
		N, 
		reps,
		K
	):
		self.N = N
		self.reps = reps
		self.K = K # transition matrix
		self.d = self.K.shape[0]

class ContractiveCircularCoupler(utils.ContractiveSampler, CircularCoupler):
	"""
	Implements a circular coupler for the contractive chain
	in Glynn and Rhee (2014).
	"""

	def __init__(self, *args):
		CircularCoupler.__init__(self, *args)

class ApprxContractiveCircularCoupler(utils.ApprxContractiveSampler, CircularCoupler):
	"""
	Implements an approximate circular coupler for the
	contractive chain in Glynn and Rhee (2014).
	"""

	def __init__(self, N, reps, dec=3):
		"""
		dec = number of decimals to round to
		"""
		self.N = N
		self.reps = reps
		self.dec = dec

class IsingCircularCoupler(utils.IsingSampler, CircularCoupler):

	def __init__(self, N, reps, D, theta):
		"""
		D : dimensionality of grid
		theta : temperature parameter
		"""
		self.N = N
		self.reps = reps
		self.D = D
		self.theta = theta