# %%
# Comparison of juliet and radvel fits for one planet
import radvel
import time
import os
import corner
import numpy as np
import matplotlib.pyplot as plt

# %%
# juliet, copied from https://juliet.readthedocs.io/en/latest/tutorials/rvfits.html
import juliet
priors = {}

# Name of the parameters to be fit:
params = ['P_p1','t0_p1','mu_CORALIE14', \
          'mu_CORALIE07','mu_HARPS','mu_FEROS',\
          'K_p1', 'ecc_p1', 'omega_p1', 'sigma_w_CORALIE14','sigma_w_CORALIE07',\
           'sigma_w_HARPS','sigma_w_FEROS']
params_map = {
    "per1": "P_p1",
    "tc1": "t0_p1",
    "k1": "K_p1",
    "gamma_CORALIE07": "mu_CORALIE07",
    "jit_CORALIE07": "sigma_w_CORALIE07",
    "gamma_CORALIE14": "mu_CORALIE14",
    "jit_CORALIE14": "sigma_w_CORALIE14",
    "gamma_FEROS": "mu_FEROS",
    "jit_FEROS": "sigma_w_FEROS",
    "gamma_HARPS": "mu_HARPS",
    "jit_HARPS": "sigma_w_HARPS",
}

# Distributions:
dists = ['normal','normal','uniform', \
         'uniform','uniform','uniform',\
         'uniform','fixed', 'fixed', 'loguniform', 'loguniform',\
         'loguniform', 'loguniform']

# Hyperparameters
P = 1.007917
P_std = 0.000073
t0 = 2458325.5386
t0_std = 0.0011
hyperps = [[P, P_std], [t0,t0_std], [-100,100], \
           [-100,100], [-100,100], [-100,100], \
           [0.,100.], 0., 90., [1e-3, 100.], [1e-3, 100.], \
           [1e-3, 100.], [1e-3, 100.]]

# Populate the priors dictionary:
for param, dist, hyperp in zip(params, dists, hyperps):
    priors[param] = {}
    priors[param]['distribution'], priors[param]['hyperparameters'] = dist, hyperp

data_file = os.path.join(radvel.DATADIR, "rvs_toi141.dat")
dataset = juliet.load(priors = priors, rvfilename=data_file, out_folder = 'toi141_rvs')

# %%
start = time.time()
juliet_results = dataset.fit(n_live_points = 300)
juliet_time = time.time() - start
print(f"Juliet took {juliet_time:.2f} sec")

# %%
juliet_vary_names = [juliet_results.model_parameters[i] for i in range(len(juliet_results.model_parameters)) if juliet_results.data.priors[juliet_results.model_parameters[i]]['distribution'].lower() != 'fixed']

# %%
# Plot HARPS and FEROS datasets in the same panel. For this, first select any
# of the two and substract the systematic velocity to get the Keplerian signal.
# Let's do it with FEROS. First generate times on which to evaluate the model:
min_time, max_time = np.min(dataset.times_rv['FEROS'])-30,\
                     np.max(dataset.times_rv['FEROS'])+30

model_times = np.linspace(min_time,max_time,1000)

# Now evaluate the model in those times, and substract the systemic-velocity to
# get the Keplerian signal:
keplerian = juliet_results.rv.evaluate('FEROS', t = model_times) - \
            np.median(juliet_results.posteriors['posterior_samples']['mu_FEROS'])

# Now plot the (systematic-velocity corrected) RVs:
fig = plt.figure(figsize=(12,5))
instruments = ['FEROS','HARPS']
colors = ['cornflowerblue','orangered']
for i in range(len(instruments)):
    instrument = instruments[i]
    # Evaluate the median jitter for the instrument:
    jitter = np.median(juliet_results.posteriors['posterior_samples']['sigma_w_'+instrument])
    # Evaluate the median systemic-velocity:
    mu = np.median(juliet_results.posteriors['posterior_samples']['mu_'+instrument])
    # Plot original data with original errorbars:
    plt.errorbar(dataset.times_rv[instrument]-2457000,dataset.data_rv[instrument]-mu,\
                 yerr = dataset.errors_rv[instrument],fmt='o',\
                 mec=colors[i], ecolor=colors[i], elinewidth=3, mfc = 'white', \
                 ms = 7, label=instrument, zorder=10)

    # Plot original errorbars + jitter (added in quadrature):
    plt.errorbar(dataset.times_rv[instrument]-2457000,dataset.data_rv[instrument]-mu,\
                 yerr = np.sqrt(dataset.errors_rv[instrument]**2+jitter**2),fmt='o',\
                 mec=colors[i], ecolor=colors[i], mfc = 'white', label=instrument,\
                 alpha = 0.5, zorder=5)

# Plot Keplerian model:
plt.plot(model_times-2457000, keplerian,color='black',zorder=1)
plt.ylabel('RV (m/s)')
plt.xlabel('Time (BJD - 2457000)')
plt.title('1 Planet Fit | Log-evidence: {0:.3f} $\pm$ {1:.3f}'.format(juliet_results.posteriors['lnZ'],\
       juliet_results.posteriors['lnZerr']))
plt.ylim([-20,20])
plt.xlim([1365,1435])
plt.show()

# %%
# RadVel
from pandas import read_csv

# Load the data in a nicer format
data_df = read_csv(os.path.join(radvel.DATADIR, "rvs_toi141.dat"), sep=" ", names=["t", "rv", "erv", "inst"])
t, vel, errvel, inst = data_df.t.values, data_df.rv.values, data_df.erv.values, data_df.inst.values

#
# %%
radvel_params = radvel.Parameters(1, basis="per tc e w k")
radvel_params["per1"] = radvel.Parameter(value=P)
radvel_params["tc1"] = radvel.Parameter(value=t0)
radvel_params["e1"] = radvel.Parameter(value=0.0, vary=False)
radvel_params["w1"] = radvel.Parameter(value=90.0 * np.pi / 180.0, vary=False)
radvel_params["k1"] = radvel.Parameter(value=10.0)
radvel_params['dvdt'] = radvel.Parameter(value=0.,vary=False)
radvel_params['curv'] = radvel.Parameter(value=0.,vary=False)

model = radvel.RVModel(radvel_params)

# %%
likes = []
inst_groups = data_df.groupby("inst").groups
inst_names = list(inst_groups.keys())
for inst in inst_names:
    indices = inst_groups[inst]
    like_inst = radvel.RVLikelihood(model, t[indices], vel[indices], errvel[indices], suffix=f"_{inst}")
    like_inst.params['gamma_'+inst] = radvel.Parameter(value=np.mean(vel[indices]), vary=True)
    like_inst.params['jit_'+inst] = radvel.Parameter(value=np.mean(errvel[indices]), vary=True)
    likes.append(like_inst)

like = radvel.CompositeLikelihood(likes)
print(like)

# %%
post = radvel.posterior.Posterior(like)

post.priors += [radvel.prior.Gaussian("per1", P, P_std)]
post.priors += [radvel.prior.Gaussian("tc1", t0, t0_std)]
post.priors += [radvel.prior.HardBounds("gamma_CORALIE14", -100.0, 100.0)]
post.priors += [radvel.prior.HardBounds("gamma_CORALIE07", -100.0, 100.0)]
post.priors += [radvel.prior.HardBounds("gamma_HARPS", -100.0, 100.0)]
post.priors += [radvel.prior.HardBounds("gamma_FEROS", -100.0, 100.0)]
post.priors += [radvel.prior.HardBounds("k1", 0.0, 100.0)]
post.priors += [radvel.prior.Jeffreys("jit_CORALIE14", 1e-3, 100.0)]
post.priors += [radvel.prior.Jeffreys("jit_CORALIE07", 1e-3, 100.0)]
post.priors += [radvel.prior.Jeffreys("jit_HARPS", 1e-3, 100.0)]
post.priors += [radvel.prior.Jeffreys("jit_FEROS", 1e-3, 100.0)]
print(post)

# %%
for pname in params:
    if juliet_results.data.priors[pname]['distribution'] == 'fixed':
        juliet_results.posteriors[pname] = juliet_results.data.priors[pname][
            'hyperparameters']
med_params = np.median(juliet_results.posteriors["posterior_samples"]["unnamed"], axis=0)
juliet_logl = juliet_results.loglike(med_params)
print(juliet_logl)
juliet_results.rv.get_log_likelihood(juliet_results.posteriors)
for instrument in juliet_results.rv.inames:
    residuals = juliet_results.rv.data[instrument] - juliet_results.rv.model[instrument]["deterministic"]
    ll = juliet_results.rv.gaussian_log_likelihood(
        residuals,
        juliet_results.rv.model[instrument]['deterministic_variances'])
    print(radvel.likelihood.loglike_jitter(residuals, juliet_results.rv.model[instrument]['deterministic_variances']**0.5, 0))
    print(ll)

# %%
med_params_dict = dict(zip(juliet_vary_names, med_params))
med_params_radvel = [med_params_dict[params_map[pname]] for pname in post.name_vary_params()]
post.logprob_array(med_params_radvel)
print(post.likelihood_ns_array(med_params_radvel))
assert post.likelihood_ns_array(med_params_radvel) == post.likelihood.logprob_array(med_params_radvel)


# %%
n_params = len(post.name_vary_params())
n_prior_samples = 10_000
rng = np.random.default_rng()
u = rng.uniform(size=(n_prior_samples, n_params))
radvel_prior_samples = post.prior_transform(u.T).T
juliet_prior_samples = np.empty_like(u)
for i in range(u.shape[0]):
    juliet_prior_samples[i] = juliet_results.prior_transform_r(u[i])

juliet_prior_samples_dict = dict(zip(juliet_vary_names, juliet_prior_samples.T))
juliet_prior_samples_radvel = np.empty_like(juliet_prior_samples)
for i, radvel_name in enumerate(post.name_vary_params()):
    juliet_name = params_map[radvel_name]
    print(i, radvel_name, juliet_name)
    juliet_prior_samples_radvel[:, i] = juliet_prior_samples_dict[juliet_name]
fig = corner.corner(radvel_prior_samples, labels=post.name_vary_params(), show_titles=True, title_fmt=".7f")
corner.corner(juliet_prior_samples_radvel, labels=post.name_vary_params(), fig=fig, color="b")
plt.show()

# %%
corner.corner(radvel_prior_samples, labels=post.name_vary_params(), show_titles=True, title_fmt=".7f")
corner.corner(juliet_prior_samples_radvel, labels=post.name_vary_params(), show_titles=True, title_fmt=".7f", color="b")
plt.show()


# %%
from radvel import nested_sampling

start = time.time()
radvel_results = nested_sampling.run_multinest(post, run_kwargs={"n_live_points": 300, "outputfiles_basename": "toi141_radvel/out"})
radvel_time = time.time() - start
print(f"Radvel took {radvel_time:.2f} sec")

# %%
print(f"RadVel evidence: {radvel_results['lnZ']} +/- {radvel_results['lnZerr']}")
print(f"Juliet evidence: {juliet_results.posteriors['lnZ']} +/- {juliet_results.posteriors['lnZerr']}")

# %%
juliet_samples_dict = juliet_results.posteriors["posterior_samples"]
juliet_samples = np.empty((len(juliet_samples_dict["P_p1"]), n_params))
for i, radvel_name in enumerate(post.name_vary_params()):
    juliet_name = params_map[radvel_name]
    juliet_samples[:, i] = juliet_samples_dict[juliet_name]
radvel_samples = radvel_results["samples"][:, :-1]
fig = corner.corner(radvel_samples, labels=post.name_vary_params())
corner.corner(juliet_samples, fig=fig, color="b")
plt.show()
