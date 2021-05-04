import numpy as np
from scipy import stats
import cvxpy as cp
####################### Generic Utility Functions ############################

def generate_discrete_chain(d, seed=1234):
	"""
	Utility function, creates random discrete markov chain
	d : integer
		size of state space
	"""
	np.random.seed(seed)

	# Generate random stochastic matrix
	K = np.eye(d)
	K = np.random.uniform(size=(d, d))
	K = K / K.sum(axis=1, keepdims=True)

	# Find invariant distribution pi (using convex solver)
	pi = cp.Variable(d)
	constraints = [
		cp.sum(pi) == 1,
		pi <= 1,
		pi >= 0,
		pi @ K == pi
	]
	problem = cp.Problem(objective=cp.Maximize(1), constraints=constraints)
	problem.solve()
	pi = pi.value

	# Check invariant dist is correct
	assert np.abs(pi - np.dot(pi, K)).mean() < 1e-5

	# Return
	return pi, K

def sample_discrete_rvs(pi0, reps, seed):
	"""
	pi0 : distribution to sample from, of shape (d, reps) or just (d,)
	reps : number of samples
	seed : random seeed
	"""
	# Cumulative sum
	cumpi0 = np.cumsum(pi0, axis=0)
	
	# Sample uniforms and output
	np.random.seed(seed)
	unifs = np.random.uniform(size=(1, reps))
	
	# If 1-dimensional pi0, reshape to make 2d
	if (len(cumpi0.shape) == 1):
		cumpi0 = cumpi0.reshape(-1, 1)
		
	flags = unifs <= cumpi0 # d x reps
	return np.argmax(flags, axis=0)

####################### Base classes which do MCMC sampling ######################

class DiscreteSampler():
	""" Samples from discrete markov chain """

	def pi0(self, seed=None):
		""" samples from discrete uniform dist """
		return sample_discrete_rvs(
			pi0=np.ones((self.d))/self.d,
			reps=self.reps,
			seed=seed
		).reshape(1, -1)

	def sample_next(self, seed, xprev, yprev=None):
		""" samples from transition matrix K """
		reps = xprev.shape[-1] # could be different from self.reps
		return sample_discrete_rvs(
			self.K[xprev[0].astype(int)].T, reps, seed=seed
		).reshape(1, reps)

class ContractiveSampler():
	""" Samples from the contractive chain from Glynn and Rhee (2014) """

	def pi0(self):
		return stats.beta.rvs(a=1, b=5, size=(1, self.reps))

	def sample_next(self, seed, xprev, yprev=None):
		np.random.seed(seed)
		reps = xprev.shape[-1]
		return xprev / 2.0 + np.random.binomial(1, 0.5, size=(1, reps))

class ApprxContractiveSampler():
	""" Samples from a rounded version of the contractive chain from Glynn and Rhee (2014) """

	def pi0(self):
		x0s = stats.beta.rvs(a=1, b=5, size=(1, self.reps))
		return np.around(x0s, self.dec)

	def sample_next(self, seed, xprev, yprev=None):
		np.random.seed(seed)
		reps = xprev.shape[-1]
		# Add noise to previous x
		width = 10**(-self.dec)
		xprevtilde = xprev + np.random.uniform(
			low=-width/2, high=width/2, size=(1, reps)
		)
		# Follow the initial chain
		xnext = xprevtilde/2.0 + np.random.binomial(1, 0.5, size=(1, reps))
		# Round
		return np.around(xnext, self.dec)

class IsingSampler():
	""" samples from ising model """

	def pi0(self):
		x0s = 1 - 2*np.random.binomial(1, 0.5, size=(self.D, self.D, self.reps))
		return x0s.reshape(-1, self.reps)

	def sample_next(self, seed, xprev, yprev=None):
		"""
		Gibbs sample
		"""
		# Set seed and generate order for gibbs samples
		xprev = xprev.reshape(self.D, self.D, -1)
		reps = xprev.shape[-1]
		np.random.seed(seed)
		xinds = np.arange(self.D)
		np.random.shuffle(xinds)
		yinds = np.arange(self.D)
		np.random.shuffle(yinds)
		uniforms = np.random.uniform(size=(self.D, self.D, reps))		
		
		# Initialize output
		xnext = xprev.copy()
		
		# Loop through and do Gibbs sample
		for i in xinds:
			for j in yinds:
				sni = xnext[(i + 1) % self.D, j] +\
					  xnext[i, (j + 1) % self.D] +\
					  xnext[(i - 1) % self.D, j] +\
					  xnext[i, (j - 1) % self.D]
				prob1 = np.exp(self.theta * sni) 
				prob1 = prob1 / (prob1 + np.exp(-self.theta * sni))
				flags = uniforms[i, j] <= prob1
				xnext[i, j] = 2*flags - 1 
				
		return xnext.reshape(-1, reps)

def ising_statistic(x):
	"""
	x : (N, D^2, reps)
	"""
	N = x.shape[0]
	d = int(np.sqrt(x.shape[1]))
	if d <= 2:
		raise ValueError(f"dimension ({d}) must be > 2")
	reps = x.shape[2]
	x = x.reshape(N, d, d, reps)
	# Shifting for ising statistics
	shift_left = x[:, np.arange(1, d+1) % d, :, :]
	shift_up = x[:, :, np.arange(1, d+1) % d, :]
	# compute
	return np.sum(x * shift_left + x * shift_up, axis=(1,2))









