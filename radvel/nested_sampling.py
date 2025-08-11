import os
import shutil
from typing import Optional

from radvel.posterior import Posterior


def run_dynesty(
    post: Posterior,
    sampler_type: str = "static",
    sampler_kwargs: Optional[dict] = None,
    run_kwargs: Optional[dict] = None,
) -> dict:
    from dynesty import DynamicNestedSampler, NestedSampler

    run_kwargs = run_kwargs or {}
    sampler_kwargs = sampler_kwargs or {}

    if sampler_type == "static":
        sampler_class = NestedSampler
    elif sampler_type == "dynamic":
        sampler_class = DynamicNestedSampler
    else:
        raise ValueError(
            f"Expected 'dynamic' or 'static' as sampler_type. Got {sampler_type}"
        )

    resume = run_kwargs.get("resume", False)
    output_file = run_kwargs.get("checkpoint_file", None)

    if resume and output_file is not None and os.path.exists(output_file):
        sampler = sampler_class.restore(output_file)
    else:
        sampler = sampler_class(
            post.likelihood_ns_array,
            post.prior_transform,
            len(post.name_vary_params()),
            **sampler_kwargs,
        )
        if resume and output_file is not None and not os.path.exists(output_file):
            run_kwargs["resume"] = False

    if output_file is not None and not os.path.exists(output_file):
        outdir = os.path.dirname(output_file)
        os.makedirs(outdir)
    sampler.run_nested(**run_kwargs)
    if output_file:
        sampler.save(output_file)

    results = {
        "samples": sampler.results.samples_equal(),
        "lnZ": sampler.results["logz"][-1],
        "lnZerr": sampler.results["logzerr"][-1],
        "sampler": sampler,
    }

    return results


def run_ultranest(
    post: Posterior,
    sampler_kwargs: Optional[dict] = None,
    run_kwargs: Optional[dict] = None,
) -> dict:
    from ultranest import ReactiveNestedSampler

    run_kwargs = run_kwargs or {}
    sampler_kwargs = sampler_kwargs or {}

    sampler = ReactiveNestedSampler(
        post.name_vary_params(),
        post.likelihood_ns_array,
        post.prior_transform,
        **sampler_kwargs,
    )

    sampler.run(**run_kwargs)

    results = {
        "samples": sampler.results["samples"],
        "lnZ": sampler.results["logz"],
        "lnZerr": sampler.results["logzerr"],
        "sampler": sampler,
    }

    return results


def run_multinest(
    post: Posterior, overwrite: bool = False, run_kwargs: Optional[dict] = None
) -> dict:
    import pymultinest

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
    results["samples"] = a.get_equal_weighted_posterior()[:, :-1]
    results["lnZ"] = a.get_stats()["global evidence"]
    results["lnZerr"] = a.get_stats()["global evidence error"]

    if tmp:
        shutil.rmtree(outdir)

    return results


def run_nautilus(
    post: Posterior,
    sampler_kwargs: Optional[dict] = None,
    run_kwargs: Optional[dict] = None,
) -> dict:
    from nautilus import Sampler

    sampler_kwargs = sampler_kwargs or {}
    run_kwargs = run_kwargs or {}

    ndim = len(post.name_vary_params())
    sampler = Sampler(
        post.prior_transform, post.likelihood_ns_array, n_dim=ndim, **sampler_kwargs
    )
    sampler.run(**run_kwargs)
    results = {
        "samples": sampler.posterior(equal_weight=True)[0],
        "lnZ": sampler.log_z,
        "lnZerr": sampler.n_eff**-0.5,
        "sampler": sampler,
    }
    return results


BACKENDS = {
    "dynesty-static": run_dynesty,
    "dynesty-dynamic": run_dynesty,
    "multinest": run_multinest,
    "ultranest": run_ultranest,
    "nautilus": run_nautilus,
}


def run(
    post: Posterior,
    output_directory: Optional[str] = None,
    overwrite: bool = False,
    sampler: str = "ultranest",
    run_kwargs: Optional[dict] = None,
    sampler_kwargs: Optional[dict] = None,
) -> dict:
    post.check_proper_priors()

    if output_directory is not None:
        # TODO: Handle output and overwrite stuff here
        pass

    sampler = sampler.lower()
    if sampler == "pymultinest":
        sampler = "multinest"

    # fmt: off
    if sampler == "ultranest":
        results = run_ultranest(post, sampler_kwarg=sampler_kwargs, run_kwargs=run_kwargs)
    elif sampler == "dynesty-static":
        results = run_dynesty(post, sampler_type="static", sampler_kwargs=sampler_kwargs, run_kwargs=run_kwargs)
    elif sampler == "dynesty-dynamic":
        results = run_dynesty(post, sampler_type="dynamic", sampler_kwargs=sampler_kwargs, run_kwargs=run_kwargs)
    elif sampler == "multinest":
        if sampler_kwargs is not None:
            raise TypeError("Argument sampler_kwargs is invalid for sampler 'multinest', only run_kwargs is supported")
        results = run_multinest(post, overwrite=overwrite, run_kwargs=run_kwargs)
    elif sampler == "nautilus":
        results = run_nautilus(post, sampler_kwarg=sampler_kwargs, run_kwargs=run_kwargs)
    else:
        raise ValueError(f"Unknown sampler '{sampler}'. Available options are {list(BACKENDS.keys())}")
    # fmt: on

    return results
