import os
import shutil
from typing import Optional

# TODO: Make libraries optional
import pymultinest
from dynesty import DynamicNestedSampler, NestedSampler
from nautilus import Sampler
from ultranest import ReactiveNestedSampler

from radvel.posterior import Posterior


def run_dynesty(
    post: Posterior,
    sampler_type: str = "static",
    sampler_kwargs: Optional[dict] = None,
    run_kwargs: Optional[dict] = None,
) -> NestedSampler | DynamicNestedSampler:
    run_kwargs = run_kwargs or {}
    sampler_kwargs = sampler_kwargs or {}

    post.check_proper_priors()

    if sampler_type == "static":
        sampler = NestedSampler
    elif sampler_type == "dynamic":
        sampler = DynamicNestedSampler
    else:
        raise ValueError(
            f"Expected 'dynamic' or 'static' as sampler_type. Got {sampler_type}"
        )

    sampler = sampler(
        post.likelihood_ns_array,
        post.prior_transform,
        len(post.name_vary_params()),
        **sampler_kwargs,
    )
    sampler.run_nested(**run_kwargs)

    return sampler


def run_ultranest(
    post: Posterior,
    sampler_kwargs: Optional[dict] = None,
    run_kwargs: Optional[dict] = None,
) -> ReactiveNestedSampler:
    run_kwargs = run_kwargs or {}
    sampler_kwargs = sampler_kwargs or {}
    post.check_proper_priors()

    sampler = ReactiveNestedSampler(
        post.name_vary_params(),
        post.likelihood_ns_array,
        post.prior_transform,
        **sampler_kwargs,
    )

    sampler.run(**run_kwargs)

    return sampler


def run_multinest(
    post: Posterior, overwrite: bool = False, run_kwargs: Optional[dict] = None
) -> dict:
    run_kwargs = run_kwargs or {}

    if "outputfiles_basename" in run_kwargs:
        outname = run_kwargs["outputfiles_basename"]
        tmp = False
    else:
        outname = "tmpdir/out"
        run_kwargs["outputfiles_basename"] = outname
        tmp = True
        overwrite = True

    resume = run_kwargs.get("resume", True)

    outdir = os.path.dirname(outname)
    os.makedirs(outdir, exist_ok=overwrite or resume)

    def loglike(p, ndim, nparams):
        # This is required to avoid segfault
        # See here: https://github.com/JohannesBuchner/PyMultiNest/issues/41, which I semi-understand
        p = [p[i] for i in range(ndim)]
        return post.likelihood_ns_array(p)

    def prior_transform(u, ndim, nparams):
        post.prior_transform(u, inplace=True)

    ndim = len(post.name_vary_params())

    pymultinest.run(loglike, prior_transform, ndim, **run_kwargs)

    a = pymultinest.Analyzer(outputfiles_basename=outname, n_params=ndim)

    results = {}
    results["samples"] = a.get_equal_weighted_posterior()
    results["lnZ"] = a.get_stats()["global evidence"]
    results["lnZerr"] = a.get_stats()["global evidence error"]

    if tmp:
        shutil.rmtree(outname)

    return results


def run_nautilus(
    post: Posterior,
    sampler_kwargs: Optional[dict] = None,
    run_kwargs: Optional[dict] = None,
) -> Sampler:
    sampler_kwargs = sampler_kwargs or {}
    run_kwargs = run_kwargs or {}

    ndim = len(post.name_vary_params())
    sampler = Sampler(
        post.prior_transform, post.likelihood_ns_array, n_dim=ndim, **sampler_kwargs
    )
    sampler.run(**run_kwargs)
    return sampler


BACKENDS = {
    "dynesty-static": run_dynesty,
    "dynesty-dynamic": run_dynesty,
    "multinest": run_multinest,
    "ultranest": run_ultranest,
    "nautilus": run_nautilus,
}
