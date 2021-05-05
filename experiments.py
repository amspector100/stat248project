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
TCOLOR = "red"
TLABEL = "true value"
FIGSIZE = (10, 6)

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

def compute_bootstrapped_ests(iid_ests):
	"""
	iid_ests : shape reps x batch
	"""
	final_ests = []
	n = iid_ests.shape[0]
	for _ in range(n):
		new_ests = iid_ests[np.random.randint(0, n, size=(n,))]
		#print("here", iid_ests.shape, new_ests.shape)
		final_ests.append(new_ests.mean(axis=0))
	
	return np.array(final_ests) 


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
	return [mean_est, se, se_of_se, lower, upper]

def compare_methods(
	UClass, # umcmc class
	CClass, # circular coupler class
	N, # number of iterations
	reps, # replications
	setting, # name of setting for plots
	savename, # where to save these things
	hs=[lambda x: x], # list of functions
	truevals=[None], # list of true values
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
	for h, trueval in zip(hs, truevals):
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
		#shift0 = trueval if trueval is not None else 0
		#name0 = "Estimator Value" if shift0 == 0 else "Bias"
		for i, name, fname in zip(
			[0, 1],
			["Estimator Value", "Standard Error"],
			["estval_k.png", "se_k.png"],
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
			if trueval is not None:
				ccoupler_ests = h(ccoupler.ys).mean(axis=0)

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
				#title=f"{name} for Various Methods in {setting}",
				xlabel="Proportion of Sample Discarded",
				ylabel=name,
			)
			plt.savefig(f"{output_dir}{fname}", dpi=500, bbox_inches='tight')
			plt.show()

		if trueval is not None:
			# Two cases: one where we use all reps, one where we use just one
			c_iid_ests = h(ccoupler.ys).reshape(N, reps).mean(axis=0)
			b_iid_ests = burnin_ests
			u_iid_ests = umcmc_ests
			# Order: b, c, u
			mse_stats_1 = []
			sq_diffs_1 = [None, None, None]
			for iid_ests in [b_iid_ests, c_iid_ests, u_iid_ests]:
				sq_diffs = np.power(iid_ests - trueval, 2).reshape(-1, reps)
				mse_stats_1.append(compute_mean_SE(
					sq_diffs.T.reshape(1, reps, -1)
				))
			# Case 2: use all reps
			mse_stats_all = []
			sq_diffs_all = []
			for iid_ests in [b_iid_ests.T, c_iid_ests, u_iid_ests.T]:
				new_ests = compute_bootstrapped_ests(iid_ests) # reps x batch
				sq_diffs = np.power(new_ests - trueval, 2).reshape(1, reps, -1) # reps x batch
				sq_diffs_all.append(sq_diffs)
				mse_stats_all.append(compute_mean_SE(sq_diffs))

			for mse_stats, mse_type, sq_diffs, fadd, scale_se in zip(
				[mse_stats_1, mse_stats_all],
				["(Single Est.)", "(Combined Est.)"],
				[sq_diffs_1, sq_diffs_all],
				['_single', '_combined'],
				[1, np.sqrt(reps)],
			):
				fig, ax = plt.subplots()
				for x, stats, diffs, color, label in zip(
					[xb, xu, xu],
					mse_stats,
					sq_diffs,
					[BCOLOR, CCOLOR, UCOLOR],
					[BLABEL, CLABEL, ULABEL],
				):
					if stats[0].shape[0] == 1:
						stats[0] = np.zeros((x.shape[0])) + stats[0]
						stats[1] = np.zeros((x.shape[0])) + stats[1]
					ax.plot(
						x,
						stats[0][0:x.shape[0]],
						color=color,
						label=label,
					)

					if fadd == '_single':
						for coef in [-2, 2]:
							ax.plot(
								x, 
								stats[0][0:x.shape[0]] + scale_se*coef*stats[1][0:x.shape[0]], 
								color=color, 
								linestyle='dotted',
							)
					else:
						diffs = diffs.reshape(reps, -1)
						for q in [0.025, 0.975]:
							yvals = np.quantile(diffs, q, axis=0)
							if yvals.shape[0] == 1:
								yvals = np.zeros((x.shape[0])) + yvals
							ax.plot(
								x, 
								yvals[0:x.shape[0]], 
								color=color, 
								linestyle='dotted',
							)

				ax.legend()
				ax.set(
					xlabel="Proportion of Sample Discarded",
					ylabel=f"Mean-Squared Error {mse_type}",
				)
				plt.savefig(f"{output_dir}MSE_k_{fadd}.png", dpi=500, bbox_inches='tight')
				plt.show()


	# Plot means/variances for the three methods varying m/N
	for h, trueval in zip(hs, truevals):
		# Estimates for different Ns for circular coupling
		min_N = max(2, int(0.1*N))
		N_vals = np.linspace(min_N, N, 10).astype(np.int32)
		ccoupler_stats_unformatted = []
		conv_pcts = []
		c_means = []
		for new_N in N_vals:
			print(f"Recomputing ccoupler for N={new_N}")
			# Run circular coupler
			new_ccoupler = CClass(N=new_N, reps=reps, **kwargs)
			new_ccoupler.forward()
			new_ccoupler.backward()
			# Percent converged
			new_ctimes = new_ccoupler.convergence_times()
			new_prop_converged = 1 - np.isnan(new_ctimes).sum() / reps
			conv_pcts.append(np.around(100*new_prop_converged, 1))
			# Compute statistics
			new_c_stats = h(new_ccoupler.ys).reshape(new_N, reps)
			c_means.append(new_c_stats.mean(axis=0))
			ccoupler_stats_unformatted.append(compute_mean_SE(
				new_c_stats
			))

		# Reorganize statistics into the old format
		ccoupler_stats = [[] for _ in range(len(burnin_stats))]
		for x in ccoupler_stats_unformatted:
			for k in range(len(x)):
				ccoupler_stats[k].append(x[k])
		for j in range(len(ccoupler_stats)):
			ccoupler_stats[j] = np.array(ccoupler_stats[j])

		# Estimates for different M for burn-in/UMCMC, much easier
		u_ests = ucoupler.compute_Hkm_diff_m(h=h, min_m=min_N)
		u_stats = compute_mean_SE(
			u_ests.T.reshape(1, reps, -1)
		) 
		b_ests = ucoupler.compute_burnin_diff_m(h=h, min_m=min_N)
		b_stats = compute_mean_SE(
			b_ests.T.reshape(1, reps, -1)
		)

		# Plot convergence values
		b_conv_pct = np.around(100*((
			ucoupler.taus.reshape(reps, 1) <= np.arange(min_N, N).reshape(1, -1)
		).mean(axis=0)), 1)
		c_conv_pct = np.array(conv_pcts)
		fig, ax = plt.subplots(figsize=FIGSIZE)
		ax.plot(np.arange(min_N, N), b_conv_pct, color=BCOLOR, label=BLABEL)
		ax.plot(N_vals, c_conv_pct, color=CCOLOR, label=CLABEL)
		ax.legend()
		plt.savefig(f"{output_dir}convtimes_N.png")


		for i, name, fname in zip(
			[0, 1],
			["Estimator Value", "Standard Error"],
			["estval_N.png", "se_N.png"],
		):

			# Create plot
			fig, ax = plt.subplots(figsize=(10,6))

			# Create x-axes
			mb = b_stats[i].shape[0]
			xb = np.arange(min_N, mb+min_N)
			mu = u_stats[i].shape[0]
			xu = np.arange(min_N, mu+min_N)
			xc = N_vals

			for x, stats, color, label in zip(
				[xb, xu, xc],
				[b_stats, u_stats, ccoupler_stats],
				[BCOLOR, UCOLOR, CCOLOR],
				[BLABEL, ULABEL, CLABEL],
			):
				ax.plot(
					x,
					stats[i],
					color=color,
					label=label,
				)
				for coef in [-2, 2]:
					ax.plot(
						x, 
						stats[i] + coef*stats[i+1], 
						color=color, 
						linestyle='dotted',
					)
			ax.legend()
			ax.set(
				xlabel="Number of Samples (N or m)",
				ylabel=name,
			)
			plt.savefig(f"{output_dir}{fname}", dpi=500, bbox_inches='tight')
			plt.show()

		if trueval is not None:
			# Two cases: one where we use all reps, one where we use just one
			c_iid_ests = np.array(c_means)
			b_iid_ests = b_ests
			u_iid_ests = u_ests
			# Order: b, c, u
			mse_stats_1 = []
			sq_diffs_1 = [None, None, None]
			for iid_ests in [b_iid_ests, c_iid_ests, u_iid_ests]:
				sq_diffs = np.power(iid_ests - trueval, 2).reshape(-1, reps)
				mse_stats_1.append(compute_mean_SE(
					sq_diffs.T.reshape(1, reps, -1)
				))
			# Case 2: use all reps
			mse_stats_all = []
			sq_diffs_all = []
			for iid_ests in [b_iid_ests.T, c_iid_ests.T, u_iid_ests.T]:
				new_ests = compute_bootstrapped_ests(iid_ests) # reps x batch
				sq_diffs = np.power(new_ests - trueval, 2).reshape(1, reps, -1) # reps x batch
				sq_diffs_all.append(sq_diffs)
				mse_stats_all.append(compute_mean_SE(sq_diffs))

			for mse_stats, mse_type, sq_diffs, fadd, scale_se in zip(
				[mse_stats_1, mse_stats_all],
				["(Single Est.)", "(Combined Est.)"],
				[sq_diffs_1, sq_diffs_all],
				['_single', '_combined'],
				[1, np.sqrt(reps)],
			):
				fig, ax = plt.subplots()
				for x, stats, diffs, color, label in zip(
					[xb, xc, xu],
					mse_stats,
					sq_diffs,
					[BCOLOR, CCOLOR, UCOLOR],
					[BLABEL, CLABEL, ULABEL],
				):
					# if stats[0].shape[0] == 1:
					# 	stats[0] = np.zeros((x.shape[0])) + stats[0]
					# 	stats[1] = np.zeros((x.shape[0])) + stats[1]
					ax.plot(
						x,
						stats[0],
						color=color,
						label=label,
					)
					if fadd == '_single':
						for coef in [-2, 2]:
							ax.plot(
								x, 
								stats[0] + scale_se*coef*stats[1], 
								color=color, 
								linestyle='dotted',
							)
					else:
						diffs = diffs.reshape(reps, -1)
						for q in [0.025, 0.975]:
							yvals = np.quantile(diffs, q, axis=0)
							ax.plot(
								x, 
								yvals, 
								color=color, 
								linestyle='dotted',
							)

				ax.legend()
				ax.set(
					xlabel="Number of Samples (N or m)",
					ylabel=f"Mean-Squared Error {mse_type}",
				)
				plt.savefig(f"{output_dir}MSE_n_{fadd}.png", dpi=500, bbox_inches='tight')
				plt.show()
