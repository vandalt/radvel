import numpy as np
import pandas as pd
import os
import radvel

"""
This config file uses Gaussian Process (GP) regression to model stellar activity.

Don't start here. Make sure you understand how RadVel works using the basic configuration
files (radvel/example_planets/epic203771098.py and radvel/example_planets/HD164922.py)
before coming back to this configuration file.
"""

# Data from Dai et al. (2017)
instnames = ['harps-n','pfs']  # no spaces in instrument names
data = pd.read_csv(os.path.join(radvel.DATADIR,'k2-131.txt'), sep=' ')
t = np.array(data['time'])
vel = np.array(data['mnvel'])
errvel = np.array(data['errvel'])
telgrps = data.groupby('tel').groups
bjd = 0.

starname = 'k2-131'
ntels = len(instnames)
planet_letters = {1: 'b'}

nplanets = 1
fitting_basis = 'per tc secosw sesinw k'

# Numbers for priors used in Dai et al. (2017)
gp_explength_mean = 9.5*np.sqrt(2.) # sqrt(2)*tau in Dai et al. (2017) [days]
gp_explength_unc = 1.0*np.sqrt(2.)
gp_perlength_mean = np.sqrt(1./(2*3.32)) # sqrt(1/2*gamma) in Dai et al. (2017)
gp_perlength_unc = 0.04

# Defensive checks for GP parameters to prevent infs/NaNs
assert gp_explength_mean > 0 and not np.isnan(gp_explength_mean) and not np.isinf(gp_explength_mean), "Invalid gp_explength_mean"
assert gp_explength_unc > 0 and not np.isnan(gp_explength_unc) and not np.isinf(gp_explength_unc), "Invalid gp_explength_unc"
assert gp_perlength_mean > 0 and not np.isnan(gp_perlength_mean) and not np.isinf(gp_perlength_mean), "Invalid gp_perlength_mean"
assert gp_perlength_unc > 0 and not np.isnan(gp_perlength_unc) and not np.isinf(gp_perlength_unc), "Invalid gp_perlength_unc"
"""
NOTE: this prior isn't equivalent to the one Dai et al. (2017) use. However,
our formulation of the quasi-periodic kernel explicitly keeps the covariance
matrix postitive semi-definite, so we use this instead. The orbit model
results aren't affected.
"""

gp_per_mean = 9.64 # T_bar in Dai et al. (2017) [days]
gp_per_unc = 0.12
Porb = 0.3693038 # [days]
Porb_unc = 0.0000091
Tc = 2457582.9360 # [BJD]
Tc_unc = 0.0011

# Defensive checks for orbital parameters
assert gp_per_mean > 0 and not np.isnan(gp_per_mean) and not np.isinf(gp_per_mean), "Invalid gp_per_mean"
assert gp_per_unc > 0 and not np.isnan(gp_per_unc) and not np.isinf(gp_per_unc), "Invalid gp_per_unc"
assert Porb > 0 and not np.isnan(Porb) and not np.isinf(Porb), "Invalid Porb"
assert Porb_unc > 0 and not np.isnan(Porb_unc) and not np.isinf(Porb_unc), "Invalid Porb_unc"
assert not np.isnan(Tc) and not np.isinf(Tc), "Invalid Tc"
assert Tc_unc > 0 and not np.isnan(Tc_unc) and not np.isinf(Tc_unc), "Invalid Tc_unc"

# Set initial parameter value guesses.
params = radvel.Parameters(nplanets,basis=fitting_basis, planet_letters=planet_letters)

params['per1'] = radvel.Parameter(value=Porb)
params['tc1'] = radvel.Parameter(value=Tc)
params['sesinw1'] = radvel.Parameter(value=0.,vary=False) # hold eccentricity fixed at 0
params['secosw1'] = radvel.Parameter(value=0.,vary=False)
params['k1'] = radvel.Parameter(value=6.55)
params['dvdt'] = radvel.Parameter(value=0.,vary=False)
params['curv'] = radvel.Parameter(value=0.,vary=False)
time_base = np.median(t)

# Define GP hyperparameters as Parameter objects.
params['gp_amp_j'] = radvel.Parameter(value=26.0)
params['gp_amp_h'] = radvel.Parameter(value=26.0)
params['gp_explength'] = radvel.Parameter(value=gp_explength_mean)
params['gp_per'] = radvel.Parameter(value=gp_per_mean)
params['gp_perlength'] = radvel.Parameter(value=gp_perlength_mean)


"""
Define a dictionary, 'hnames', specifying the names
of the GP hyperparameters corresponding to a particular
data set. Use the strings in 'instnames' to tell radvel
which data set you're talking about.
"""
hnames = {
  'harps-n': ['gp_amp_h', # GP variability amplitude
              'gp_per', # GP variability period
              'gp_explength', # GP non-periodic characteristic length
              'gp_perlength'], # GP periodic characteristic length
  'pfs': ['gp_amp_j',
          'gp_per',
          'gp_explength',
          'gp_perlength']
}

kernel_name = {'harps-n':"QuasiPer",
               'pfs':"QuasiPer"}
"""
NOTE: If all kernels are quasi-periodic, you don't need to include the
'kernel_name' lines. I included it to show you how to use different kernel
types in different likelihoods.
"""

jit_guesses = {'harps-n':2.0, 'pfs':5.3}

def initialize_instparams(tel_suffix):

    indices = telgrps[tel_suffix]

    params['gamma_'+tel_suffix] = radvel.Parameter(value=np.mean(vel[indices]))
    params['jit_'+tel_suffix] = radvel.Parameter(value=jit_guesses[tel_suffix])


for tel in instnames:
    initialize_instparams(tel)

# Add in priors (Dai et al. 2017 Section 7.2.1)
priors = [radvel.prior.Gaussian('per1', Porb, Porb_unc),
          radvel.prior.Gaussian('tc1', Tc, Tc_unc),
          radvel.prior.Jeffreys('k1', 0.01, 10.), # min and max for Jeffrey's priors estimated by Sarah
          radvel.prior.Jeffreys('gp_amp_h', 0.01, 100.),
          radvel.prior.Jeffreys('gp_amp_j', 0.01, 100.),
          radvel.prior.Jeffreys('jit_pfs', 0.01, 10.),
          radvel.prior.Jeffreys('jit_harps-n', 0.01,10.),
          # Use HardBounds instead of Gaussian for GP parameters to enforce strict bounds
          radvel.prior.HardBounds('gp_explength', 0.1, 100.),
          radvel.prior.HardBounds('gp_per', 0.1, 100.),
          radvel.prior.HardBounds('gp_perlength', 0.01, 10.0)]

