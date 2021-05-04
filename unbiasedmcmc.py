import numpy as np
from scipy import stats
import utils


################ General unbiased MCMC class #################

class UnbiasedMCMC():

	def __init__(
		self,
		m,
		reps,
		lag=1,
		**kwargs
	):

		self.m = m # min. num samples
		self.reps = reps # number of replications
		self.lag = lag

		# flags for which of the reps chains have converged
		self.converged = np.zeros((self.reps,)).astype(bool)
		self.taus = np.zeros((self.reps,))
		self.taus[:] = np.nan

		# pi0

	def pi0(self):

		raise NotImplementedError()

	def sample_next(self, xprev, yprev, seed):
		"""
		Samples Xt | Xt-1. Returns Xt.
		"""

		raise NotImplementedError()

	def sample_next_coupled(self, xprev, yprev, seed):
		"""
		Samples the next x and the next y in a coupled fashion.
		
		Returns both x and y.
		"""

		### Naive implementation, can be overridden
		xnext = self.sample_next(seed=seed, xprev=xprev, yprev=yprev)
		ynext = self.sample_next(seed=seed, xprev=yprev, yprev=xprev)
		return xnext, ynext


	def sample_all(self, max_samples=10000, verbose=False):
		"""
		Samples x, y, etc.
		"""

		self.xs = [self.pi0()]
		self.dim = self.xs[-1].shape[0]
		self.ys = [self.pi0()]
		# Sample xs in a non-coupled fashion
		for j in range(self.lag):
			self.xs.append(
				self.sample_next(xprev=self.xs[-1], yprev=None, seed=None)
			)

		# Convergence check
		self.converged = np.all(self.xs[-1]==self.ys[-1], axis=0).astype(bool)
		self.taus[(self.converged) & (np.isnan(self.taus))] = len(self.xs) - 1

		# Sample xs and ys together until convergence
		for seed in range(max_samples):

			# xnext, ynext = self.sample_next_coupled(
			# 	xprev=self.xs[-1],
			# 	yprev=self.ys[-1],
			# 	seed=seed,
			# )

			## Sample from coupled distribution for those that haven't converged
			if np.any(~self.converged):
				xnext0, ynext0 = self.sample_next_coupled(
					xprev=self.xs[-1][:, ~self.converged],
					yprev=self.ys[-1][:, ~self.converged],
					seed=seed
				)
			else:
				xnext0 = None
				ynext0 = None

			### Sample marginally for those that have converged (if any)
			if np.any(self.converged):
				xnext1 = self.sample_next(
					xprev=self.xs[-1][:, self.converged], yprev=None, seed=seed
				)
				xnext = np.zeros((self.dim, self.reps))
				xnext[:, self.converged] = xnext1
				ynext = xnext.copy()
				if xnext0 is not None:
					xnext[:, ~self.converged] = xnext0
					ynext[:, ~self.converged] = ynext0
			else:
				# If none of the chains have converged
				xnext = xnext0
				ynext = ynext0


			self.xs.append(xnext)
			self.ys.append(ynext)

			### Check for convergence
			self.converged = np.all(self.xs[-1]==self.ys[-1], axis=0).astype(bool)
			self.taus[(self.converged) & (np.isnan(self.taus))] = len(self.xs) - 1

			### Reset convergence flags
			if verbose and seed % 100 == 0 and seed != 0:
				print(f"At iteration={seed}, {100*self.converged.mean()}% of the chains are converged")
			if np.all(self.converged) and seed >= self.m:
				if verbose:
					print(f"Breaking at iteration={seed}")
				break


		self.xs = np.array(self.xs)
		self.ys = np.array(self.ys)

	def compute_Hkm(self, k, h=lambda x: x):
		"""
		h : function we are interested in computing the expectation of.
		Must be able to apply elementwise to an np.array.
		"""
		if k < self.lag:
			raise ValueError(f"k {k} must be greater than the lag {self.lag}")
		if np.any(np.isnan(self.taus)):
			raise ValueError("Some of the chains did not converge---unbiased estimation is not possible.")

		self.taus = self.taus.astype(np.int32)

		hx = h(self.xs[self.lag:]).reshape(-1, self.reps)
		hy = h(self.ys).reshape(-1, self.reps)
		diff = hx - hy
		#print("hx \n", hx)
		#print("hy \n", hy)
		#print("diff \n", diff)
		cumdiff = np.cumsum(diff, axis=0)
		#print("cumdiff \n", cumdiff)

		# Compute regular estimators
		Hkm = 0
		for j in range(k, self.m):
			Hj = hx[j-self.lag] + cumdiff[self.taus-self.lag, np.arange(self.reps)] - cumdiff[j-self.lag]
			#print(cumdiff[self.taus-1-self.lag, np.arange(self.reps)].shape)
			#print(f"Hj for j={j}", Hj, f"start={cumdiff[j-self.lag]}", f"end={cumdiff[self.taus-self.lag, np.arange(self.reps)]}")
			Hkm += Hj

		# Time-averaged estimator
		return Hkm / (self.m-k)

class DiscreteUMCMC(utils.DiscreteSampler, UnbiasedMCMC):
	""" Circular coupler for discrete markov chains """

	def __init__(
		self,
		*args,
		**kwargs
	):

		UnbiasedMCMC.__init__(self, *args, **kwargs)
		self.K = kwargs.get("K")
		self.d = self.K.shape[0]


class ContractiveUMCMC(utils.ContractiveSampler, UnbiasedMCMC):

	def __init__(self, *args):
		UnbiasedMCMC.__init__(self, *args)

class ApprxContractiveUMCMC(utils.ApprxContractiveSampler, UnbiasedMCMC):
	"""
	Implements an approximate circular coupler for the
	contractive chain in Glynn and Rhee (2014) using ideas from 
	Jacob et al. 2020.
	"""

	def __init__(self, *args, **kwargs):

		UnbiasedMCMC.__init__(self, *args, **kwargs)
		self.dec = kwargs.get("dec", 3)

class IsingUMCMC(utils.IsingSampler, UnbiasedMCMC):
	"""
	Implements an approximate circular coupler for the
	contractive chain in Glynn and Rhee (2014) using ideas from 
	Jacob et al. 2020.
	"""

	def __init__(self, *args, **kwargs):

		UnbiasedMCMC.__init__(self, *args, **kwargs)
		self.D = kwargs.get("D")
		self.theta = kwargs.get("theta")