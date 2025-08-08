# %%
import time

import radvel
import numpy as np
from pandas import read_csv
import os
from radvel import nested_sampling as rns

import matplotlib.pyplot as plt

# %%
# Set to True when to avoid re-running samplers
RESUME = False

# %%
def initialize_model(num_planets):
    time_base = 2420
    params = radvel.Parameters(num_planets,basis='per tc secosw sesinw logk') # number of planets = 2
    params['per1'] = radvel.Parameter(value=20.885258)
    params['tc1'] = radvel.Parameter(value=2072.79438)
    params['secosw1'] = radvel.Parameter(value=0.01)
    params['sesinw1'] = radvel.Parameter(value=0.01)
    params['logk1'] = radvel.Parameter(value=1.1)
    if num_planets == 2:
        params['per2'] = radvel.Parameter(value=42.363011)
        params['tc2'] = radvel.Parameter(value=2082.62516)
        params['secosw2'] = radvel.Parameter(value=0.01)
        params['sesinw2'] = radvel.Parameter(value=0.01)
        params['logk2'] = radvel.Parameter(value=1.1)
    mod = radvel.RVModel(params, time_base=time_base)
    mod.params['dvdt'] = radvel.Parameter(value=-0.02)
    mod.params['curv'] = radvel.Parameter(value=0.01)

    like = radvel.likelihood.RVLikelihood(mod, t, vel, errvel)
    like.params['gamma'] = radvel.Parameter(value=0.1, vary=False, linear=True)
    like.params['jit'] = radvel.Parameter(value=1.0)
    like.params['secosw1'].vary = False
    like.params['sesinw1'].vary = False
    like.params['per1'].vary = False
    like.params['tc1'].vary = False
    if num_planets == 2:
        like.params['secosw2'].vary = False
        like.params['sesinw2'].vary = False
        like.params['per2'].vary = False
        like.params['tc2'].vary = False

    post = radvel.posterior.Posterior(like)
    post.priors += [radvel.prior.Gaussian('jit', np.log(3), 0.5)]
    post.priors += [radvel.prior.Gaussian('logk1', np.log(5), 10)]
    post.priors += [radvel.prior.Gaussian('dvdt', 0, 1.0)]
    post.priors += [radvel.prior.Gaussian('curv', 0, 1e-1)]
    if num_planets == 2:
        post.priors += [radvel.prior.Gaussian('logk2', np.log(5), 10)]

    return post

# %%
path = os.path.join(radvel.DATADIR,'epic203771098.csv')
rv = read_csv(path)

t = np.array(rv.t)
vel = np.array(rv.vel)
errvel = rv.errvel
ti = np.linspace(rv.t.iloc[0]-5,rv.t.iloc[-1]+5,100)

# %%
post = initialize_model(2)
print(post)

# %%
def plot_results(like):
    fig = plt.figure(figsize=(12,4))
    fig = plt.gcf()
    fig.set_tight_layout(True)
    plt.errorbar(
        like.x, like.model(t)+like.residuals(), 
        yerr=like.yerr, fmt='o'
        )
    plt.plot(ti, like.model(ti))
    plt.xlabel('Time')
    plt.ylabel('RV')
    return fig

# %%
plot_results(post.likelihood)
plt.show()

# %%
multinest_start = time.time()
multinest_results = rns.run_multinest(post, run_kwargs={"outputfiles_basename": "mutlinest_demo/out", "resume": RESUME})
multinest_time = time.time() - multinest_start
print(f"Running multinest took {multinest_time / 60:.2f} min")

# %%
dynesty_start = time.time()
dynesty_sampler = dynesty_results = rns.run_dynesty(post, run_kwargs={"checkpoint_file": "dynesty_demo/dynesty.save", "resume": RESUME})
dynesty_time = time.time() - dynesty_start
print(f"Running dynesty took {dynesty_time / 60:.2f} min")

# %%
dynesty_start = time.time()
dynesty_dynamic_sampler = dynesty_results = rns.run_dynesty(post, sampler_type="dynamic", run_kwargs={"checkpoint_file": "dynesty_demo/dynesty_dynamic.save", "resume": RESUME})
dynesty_time = time.time() - dynesty_start
print(f"Running dynesty took {dynesty_time / 60:.2f} min")

# %%
ultranest_start = time.time()
ultranest_sampler = rns.run_ultranest(post, sampler_kwargs={"log_dir": "ultranest_demo", "resume": RESUME})
ultranest_time = time.time() - ultranest_start
print(f"Running ultranest took {ultranest_time / 60:.2f} min")

# %%
nautilus_start = time.time()
nautilus_sampler = rns.run_nautilus(post, sampler_kwargs={"filepath": "nautilus_demo/output.hdf5", "resume": RESUME}, run_kwargs={"verbose": True})
nautilus_time = time.time() - nautilus_start
print(f"Running nautilus took {nautilus_time / 60:.2f} min")

# %%
print(f"Multinest: {multinest_results['lnZ']:.2f} +/- {multinest_results['lnZerr']:.2f}")
print(f"Dynesty (static): {dynesty_sampler.results['logz'][-1]:.2f} +/- {dynesty_sampler.results['logzerr'][-1]:.2f}")
print(f"Dynesty (dynamic): {dynesty_dynamic_sampler.results['logz'][-1]:.2f} +/- {dynesty_dynamic_sampler.results['logzerr'][-1]:.2f}")
print(f"Ultranest: {ultranest_sampler.results['logz']:.2f} +/- {ultranest_sampler.results['logzerr']:.2f}")
print(f"Nautilus: {nautilus_sampler.log_z:.2f} +/- {nautilus_sampler.n_eff**-0.5:.2f}")

# %%
import corner

points, log_w, log_l = nautilus_sampler.posterior()
hist_kwargs = {"density": True}
fig = corner.corner(points[::10], weights=np.exp(log_w[::10]), labels=post.name_vary_params(),range=np.repeat(0.999, len(post.name_vary_params())), plot_datapoints=False, hist_kwargs=hist_kwargs)

corner.corner(multinest_results["samples"][:, :-1], color="b", fig=fig, hist_kwargs=hist_kwargs | {"color": "b"})

corner.corner(dynesty_sampler.results.samples_equal(), color="r", fig=fig, hist_kwargs=hist_kwargs | {"color": "r"},range=np.repeat(0.999, len(post.name_vary_params())))

corner.corner(dynesty_dynamic_sampler.results.samples_equal(), color="yellow", fig=fig, hist_kwargs=hist_kwargs | {"color": "yellow"},range=np.repeat(0.999, len(post.name_vary_params())))

corner.corner(ultranest_sampler.results["samples"], color="peachpuff", fig=fig, hist_kwargs=hist_kwargs | {"color": "peachpuff"}, range=np.repeat(0.999, len(post.name_vary_params())))

plt.show()
