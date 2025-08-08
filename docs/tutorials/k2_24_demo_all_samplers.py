# %%
import radvel
import numpy as np
from pandas import read_csv
import os
from radvel import nested_sampling as rns

import matplotlib.pyplot as plt


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
multinest_results = rns.run_multinest(post)

# %%
dynesty_sampler = dynesty_results = rns.run_dynesty(post)

# %%
dynesty_dynamic_sampler = dynesty_results = rns.run_dynesty(post, sampler_type="dynamic")

# %%
ultranest_sampler = rns.run_ultranest(post)

# %%
print(f"Multinest: {multinest_results['lnZ']} +/- {multinest_results['lnZerr']}")
print(f"Dynesty (static): {dynesty_sampler.results['logz'][-1]} +/- {dynesty_sampler.results['logzerr'][-1]}")
print(f"Dynesty (dynamic): {dynesty_dynamic_sampler.results['logz'][-1]} +/- {dynesty_dynamic_sampler.results['logzerr'][-1]}")
print(f"Ultranest: {ultranest_sampler.results['logz']} +/- {ultranest_sampler.results['logzerr']}")
