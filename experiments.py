import time
import sys
import os

import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import *

BCOLOR = 'orange'
BLABEL = 'Simple burn-in'
UCOLOR = 'blue'
ULABEL = "UMCMC with bias-correction"
CCOLOR = 'green'
CLABEL = "Circular coupling"

def compute_SE_of_SE(means, B=1000):
	"""
	Uses bootstrap to compute SE of SE
	"""
	ses = []
	n = means.shape[0]
	for _ in range(B):
		newmeans = means[np.random.randint(0, n, size=(n,))]
		ses.append(newmeans.std(axis=0) / np.sqrt(n))
	ses = np.array(ses)

	return np.std(ses, axis=0)


def compute_mean_SE(estimates):
	"""
	estimates : np.ndarray of shape (N, reps, batch)

	returns: mean, se, lower, upper 
	"""
	if len(estimates.shape) == 1:
		N = 1
		reps = estimates.shape[0]
		estimates = estimates.reshape(N, reps, 1)
	else:
		N = estimates.shape[0]
		reps = estimates.shape[1]
		if len(estimates.shape) == 2:
			estimates = estimates.reshape(N, reps, 1)

	means = estimates.mean(axis=0) # dim: reps x batch
	mean_est = means.mean(axis=0) # dim: (batch,)
	se = means.std(axis=0) / np.sqrt(reps)
	se_of_se = compute_SE_of_SE(means)
	lower = mean_est - 2*se 
	upper = mean_est + 2*se
	return mean_est, se, se_of_se, lower, upper

def compare_methods(
	UClass, # umcmc class
	CClass, # circular coupler class
	N, # number of iterations
	reps, # replications
	setting, # name of setting for plots
	savename, # where to save these things
	hs=[lambda x: x], # list of functions
	lag=1, 
	min_samples=None, # for burnin
	**kwargs, # kwargs for creation
):

	# Create save directory
	output_dir = f"plots/{savename}/"
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	print(f"Output directory is {output_dir}")

	# Determine min_samples from N
	min_samples = int(0.1 * N)

	# Run circular coupler
	ccoupler = CClass(N=N, reps=reps, **kwargs)
	ccoupler.forward()
	ccoupler.backward()
	ctimes = ccoupler.convergence_times()
	prop_converged = 1 - np.isnan(ctimes).sum() / reps
	pct_converged = np.around(100*prop_converged, 1)

	# Run umcmc 
	ucoupler = UClass(
		m=N, reps=reps, lag=lag, **kwargs,
	)
	ucoupler.sample_all(verbose=True)

	# Plot convergence times
	fig, ax = plt.subplots()
	sns.histplot(
		ctimes[~np.isnan(ctimes)], ax=ax, color=BCOLOR, alpha=0.5, 
		label=CLABEL + f" ({pct_converged}% converged)",
	)
	sns.histplot(
		ucoupler.taus-lag, ax=ax, color=UCOLOR, alpha=0.5, label=ULABEL,
	)
	ax.legend()
	ax.set(title=f"Convergence Times for {setting}")
	plt.savefig(f"{output_dir}convtimes.png", dpi=500, bbox_inches='tight')
	plt.show()

	# Plot means/variances for the three methods varying k
	for h in hs:
		ccoupler_stats = compute_mean_SE(
			h(ccoupler.ys).reshape(N, reps)
		) # 1d
		umcmc_ests = ucoupler.compute_all_Hkm(h=h) # batch x reps
		umcmc_stats = compute_mean_SE(
			umcmc_ests.T.reshape(1, reps, -1)
		) # m-dimensional

		burnin_ests = ucoupler.compute_burnin_ests(h=h) # m-dimensional
		burnin_stats = compute_mean_SE(
			burnin_ests.T.reshape(1, reps, -1)
		) # m-dimensional
		for i, name, fname in zip(
			[0, 1],
			["Estimator Value", "Standard Error"],
			["estval.png", "se.png"],
		):

			# Create x-axes
			mb = burnin_stats[i].shape[0] - min_samples
			xb = np.arange(1, mb+1)/mb
			mu = umcmc_stats[i].shape[0] - min_samples
			xu = np.arange(1, mu+1)/mu

			fig, ax = plt.subplots()
			ax.plot(
				xb,
				burnin_stats[i][0:-min_samples],
				color=BCOLOR,
				label=BLABEL,
			)
			ax.plot(
				xu,
				umcmc_stats[i][0:-min_samples],
				color=UCOLOR,
				label=ULABEL,
			)
			ax.plot(
				xu,
				np.zeros((xu.shape[0])) + ccoupler_stats[i], 
				color=CCOLOR, 
				label=CLABEL,
				#linetype='dashed'
			)
			ax.legend()


			for coef in [-2, 2]:
				for x, stats, color in zip(
						[xb, xu], [burnin_stats, umcmc_stats], [BCOLOR, UCOLOR]
				):
					ax.plot(
						x, 
						stats[i][0:-min_samples] + coef*stats[i+1][0:-min_samples], 
						color=color, 
						linestyle='dotted',
					)

				ax.plot(
					xu, 
					np.zeros((xu.shape[0])) + ccoupler_stats[i] + coef*ccoupler_stats[i+1],
					color=CCOLOR,
					linestyle='dotted'
				)

			ax.set(
				title=f"{name} for Various Methods in {setting}",
				xlabel="Proportion of Sample Discarded",
				ylabel=name,
			)
			plt.savefig(f"{output_dir}{fname}", dpi=500, bbox_inches='tight')
			ax.plot()

	# Plot means/variances for the three methods varying m/N
	for h in hs:
		# Estimates for different Ns for circular coupling
		min_N = max(2, int(0.1*N))
		N_vals = np.linspace(min_N, N, 10).astype(np.int32)
		ccoupler_stats_unformatted = []
		conv_pcts = []
		for new_N in N_vals:
			print(f"Recomputing ccoupler for N={new_N}")
			# Run circular coupler
			new_ccoupler = CClass(N=new_N, reps=reps, **kwargs)
			new_ccoupler.forward()
			new_ccoupler.backward()
			# Percent converged
			new_ctimes = ccoupler.convergence_times()
			new_prop_converged = 1 - np.isnan(ctimes).sum() / reps
			conv_pcts.append(np.around(100*prop_converged, 1))
			# Compute statistics
			ccoupler_stats_unformatted.append(compute_mean_SE(
				h(new_ccoupler.ys).reshape(new_N, reps)
			))
		# Reorganize statistics into the old format
		ccoupler_stats = [[] for _ in range(len(burnin_stats))]
		for x in ccoupler_stats_unformatted:
			for k in range(len(x)):
				ccoupler_stats[k].append(x[k])
		for j in range(len(ccoupler_stats)):
			ccoupler_stats[j] = np.array(ccoupler_stats[j])

		# Estimates for different M for burn-in/UMCMC, much easier
		

		# cumys = np.cumsum(ccoupler.ys, axis=0) # N x reps
		# ccoupler_ests = cumys / np.arange(1, N+1).reshape(-1, 1)
		# ccoupler_ests = ccoupler_ests[min_N:] 

		# stats = compute_mean_SE(
		# 	h(ccoupler.ys).reshape(N, reps)
		# ) # 1d
		# umcmc_ests = ucoupler.compute_all_Hkm(h=h) # batch x reps
		# umcmc_stats = compute_mean_SE(
		# 	umcmc_ests.T.reshape(1, reps, -1)
		# ) # m-dimensional

		# burnin_ests = ucoupler.compute_burnin_ests(h=h) # m-dimensional
		# burnin_stats = compute_mean_SE(
		# 	burnin_ests.T.reshape(1, reps, -1)
		# ) # m-dimensional