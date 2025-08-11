import os
import shutil
from typing import Optional

import h5py
import numpy as np

from radvel.posterior import Posterior


def run_dynesty(
    post: Posterior,
    output_dir: Optional[str] = None,
    sampler_type: str = "static",
    sampler_kwargs: Optional[dict] = None,
    run_kwargs: Optional[dict] = None,
) -> dict:
    """Run nested sampling with Dynesty
    Dynesty docs: https://dynesty.readthedocs.io/en/v2.1.5/api.html
    Args:
        post: radvel posterior object
        output_dir: Output directory where the sampler checkpoints and results will be stored. Nothing is stored by default.
            **Note**: This replaces the sampler's built-in "checkpoint_file" argument. A `dynesty.save` file is created automatically.
            When sampling is finished, the final state of the sampler is stored.
        sampler_kwargs: Dictionary of keyword arguments passed to the 'sampler' object from the underlying nested sampling package at initialization.
            See each package's documentation to learn more on the available arguments. This is not available for `sampler='multinest'`.
            Defaults to `None`.
        run_kwargs: Dictionary of keyword arguments passed to the 'run' methods from the underlying nested sampling package.
            See each package's documentation to learn more on the available aruments.
    Returns:
        Dictionary of results with the following keys:
            - `samples`: Samples dictionary with shape `(nsamples, nparams)`
            - `lnZ`: Log of the Bayesian evidence
            - `lnZ`: Statistical uncertainty on the evidence
            - `sampler`: Sampler object used by the nested sampling library. Provides more fine-grained access to the results.

    """
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

    if "checkpoint_file" in run_kwargs:
        raise ValueError(
            "checkpoint_file not supported for dynesty. Use radvel's output_dir instead."
        )
    if output_dir is not None:
        checkpoint_file = os.path.join(output_dir, "sampler.save")
        run_kwargs["checkpoint_file"] = checkpoint_file
    checkpoint_file = run_kwargs.get("checkpoint_file", None)

    if resume and checkpoint_file is not None and os.path.exists(checkpoint_file):
        sampler = sampler_class.restore(checkpoint_file)
    else:
        sampler = sampler_class(
            post.likelihood_ns_array,
            post.prior_transform,
            len(post.name_vary_params()),
            **sampler_kwargs,
        )
        # Dynesty cannot resume when the file does not exist
        if (
            resume
            and checkpoint_file is not None
            and not os.path.exists(checkpoint_file)
        ):
            run_kwargs["resume"] = False

    if checkpoint_file is not None and not os.path.exists(checkpoint_file):
        outdir = os.path.dirname(checkpoint_file)
        os.makedirs(outdir)

    sampler.run_nested(**run_kwargs)

    if checkpoint_file is not None:
        sampler.save(checkpoint_file)

    results = {
        "samples": sampler.results.samples_equal(),
        "lnZ": sampler.results["logz"][-1],
        "lnZerr": sampler.results["logzerr"][-1],
        "sampler": sampler,
    }

    return results


def run_ultranest(
    post: Posterior,
    output_dir: Optional[str] = None,
    sampler_kwargs: Optional[dict] = None,
    run_kwargs: Optional[dict] = None,
) -> dict:
    """Run nested sampling with Ultranest
    Ultranest docs: https://johannesbuchner.github.io/UltraNest/ultranest.html#ultranest.integrator.ReactiveNestedSampler
    Args:
        post: radvel posterior object
        output_dir: Output directory where the sampler checkpoints and results will be stored. Nothing is stored by default.
            **Note**: This replaces the sampler's built-in "log_dir" argument.
            The ultranest `log_dir` is automatically set to `output_dir`.
        sampler_kwargs: Dictionary of keyword arguments passed to the 'sampler' object from the underlying nested sampling package at initialization.
            See each package's documentation to learn more on the available arguments. This is not available for `sampler='multinest'`.
            Defaults to `None`.
        run_kwargs: Dictionary of keyword arguments passed to the 'run' methods from the underlying nested sampling package.
            See each package's documentation to learn more on the available aruments.
    Returns:
        Dictionary of results with the following keys:
            - `samples`: Samples dictionary with shape `(nsamples, nparams)`
            - `lnZ`: Log of the Bayesian evidence
            - `lnZ`: Statistical uncertainty on the evidence
            - `sampler`: Sampler object used by the nested sampling library. Provides more fine-grained access to the results.
    """
    from ultranest import ReactiveNestedSampler

    if "log_dir" in sampler_kwargs:
        raise ValueError(
            "log_dir not supported for ultranest. Use radvel's output_dir instead."
        )
    sampler_kwargs["log_dir"] = output_dir

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
    post: Posterior,
    output_dir: Optional[str] = None,
    overwrite: bool = False,
    run_kwargs: Optional[dict] = None,
) -> dict:
    """Run nested sampling with PyMultiNest
    PyMultiNest docs: https://johannesbuchner.github.io/PyMultiNest/pymultinest.html#
    Args:
        post: radvel posterior object
        output_dir: Output directory where the sampler checkpoints and results will be stored. Nothing is stored by default.
            **Note**: This replaces the sampler's built-in "outputfiles_basename" argument.
            If `output_dir` is specified, sets `outputfiles_basename` to `<output_dir>/out`
        overwrite: Overwrite the output files if they exist. Defaults to `False`.
        run_kwargs: Dictionary of keyword arguments passed to the 'run' methods from the underlying nested sampling package.
            See each package's documentation to learn more on the available aruments.
    Returns:
        Dictionary of results with the following keys:
            - `samples`: Samples dictionary with shape `(nsamples, nparams)`
            - `lnZ`: Log of the Bayesian evidence
            - `lnZ`: Statistical uncertainty on the evidence
    """
    import pymultinest

    run_kwargs = run_kwargs or {}

    # By default, assume we want a temporary output dir
    tmp = True

    if output_dir is None:
        output_dir = "tmpdir"
    else:
        # if an actual outupt dir was specified, it is not temporary
        tmp = False

    if "outputfiles_basename" in run_kwargs:
        raise ValueError(
            "outputfiles_basename not supported for multinest. Use radvel's output_dir instead."
        )
    run_kwargs["outputfiles_basename"] = os.path.join(output_dir, "out")

    resume = run_kwargs.get("resume", True)

    os.makedirs(output_dir, exist_ok=tmp or overwrite or resume)

    def loglike(p, ndim, nparams):
        # This is required to avoid segfault
        # See here: https://github.com/JohannesBuchner/PyMultiNest/issues/41, which I semi-understand
        p = [p[i] for i in range(ndim)]
        return post.likelihood_ns_array(p)

    def prior_transform(u, ndim, nparams):
        post.prior_transform(u, inplace=True)

    ndim = len(post.name_vary_params())

    pymultinest.run(loglike, prior_transform, ndim, **run_kwargs)

    a = pymultinest.Analyzer(
        outputfiles_basename=run_kwargs["outputfiles_basename"], n_params=ndim
    )

    results = {}
    results["samples"] = a.get_equal_weighted_posterior()[:, :-1]
    results["lnZ"] = a.get_stats()["global evidence"]
    results["lnZerr"] = a.get_stats()["global evidence error"]

    if tmp:
        shutil.rmtree(output_dir)

    return results


def run_nautilus(
    post: Posterior,
    output_dir: Optional[str] = None,
    sampler_kwargs: Optional[dict] = None,
    run_kwargs: Optional[dict] = None,
) -> dict:
    """Run nested sampling with Nautilus
    Nautilus docs: https://nautilus-sampler.readthedocs.io/en/latest/api_high.html
    Args:
        post: radvel posterior object
        output_dir: Output directory where the sampler checkpoints and results will be stored. Nothing is stored by default.
            **Note**: This replaces the sampler's built-in "filepath argument.
            The nautilus output is automatically stored in `nautilus_output.hdf5` under that location.
        sampler_kwargs: Dictionary of keyword arguments passed to the 'sampler' object from the underlying nested sampling package at initialization.
            See each package's documentation to learn more on the available arguments. This is not available for `sampler='multinest'`.
            Defaults to `None`.
        run_kwargs: Dictionary of keyword arguments passed to the 'run' methods from the underlying nested sampling package.
            See each package's documentation to learn more on the available aruments.
    Returns:
        Dictionary of results with the following keys:
            - `samples`: Samples dictionary with shape `(nsamples, nparams)`
            - `lnZ`: Log of the Bayesian evidence
            - `lnZ`: Statistical uncertainty on the evidence
            - `sampler`: Sampler object used by the nested sampling library. Provides more fine-grained access to the results.
    """
    from nautilus import Sampler

    if "filepath" in sampler_kwargs:
        raise ValueError(
            "filepath not supported for nautilus. Use radvel's output_dir instead."
        )
    if output_dir is not None:
        sampler_kwargs["filepath"] = os.path.join(output_dir, "nautilus_output.hdf5")

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
    output_dir: Optional[str] = None,
    overwrite: bool = False,
    sampler: str = "ultranest",
    sampler_kwargs: Optional[dict] = None,
    run_kwargs: Optional[dict] = None,
) -> dict:
    """Run nested sampling
    Args:
        post: radvel posterior object
        output_dir: Output directory where the sampler checkpoints and results will be stored.
            Nothing is stored by default.
            **Note**: This replaces the sampler's built-in "checkpoint_file", "log_dir", or "outputfiles_basename" argument.
            Once you specify output there, everything is saved there automatically.
            A `results.hdf5` file will also be saved with the results dict, except for the sampler.
        overwrite: Overwrite the results.hdf5 if True. This is not used for checkpoint files from the sampler as
            they can be used to resume a run. Defaults to `False`.
        sampler: name of the sampler to use. Should be one of 'ultranest', 'dynesty-static', 'dynesty-dynamic', 'nautilus', or 'multinest'.
            Defaults to 'ultranest'.
        sampler_kwargs: Dictionary of keyword arguments passed to the 'sampler' object from the underlying nested sampling package at initialization.
            See each package's documentation to learn more on the available arguments. This is not available for `sampler='multinest'`.
            Defaults to `None`.
        run_kwargs: Dictionary of keyword arguments passed to the 'run' methods from the underlying nested sampling package.
            See each package's documentation to learn more on the available aruments.
    Returns:
        Dictionary of results with the following keys:
            - `samples`: Samples dictionary with shape `(nsamples, nparams)`
            - `lnZ`: Log of the Bayesian evidence
            - `lnZ`: Statistical uncertainty on the evidence
            - `sampler`: Sampler object used by the nested sampling library. Provides more fine-grained access to the results.

    Link to each package's API documentation:
    - Ultranest: https://johannesbuchner.github.io/UltraNest/ultranest.html#ultranest.integrator.ReactiveNestedSampler
    - Nautilus: https://nautilus-sampler.readthedocs.io/en/latest/api_high.html
    - Dynesty: https://dynesty.readthedocs.io/en/v2.1.5/api.html
    - PyMultiNest: https://johannesbuchner.github.io/PyMultiNest/pymultinest.html#
    """
    post.check_proper_priors()

    if output_dir is not None:
        results_file = os.path.join(output_dir, "results.hdf5")
        if os.path.exists(results_file) and not overwrite:
            raise FileExistsError(
                f"Results file {results_file} exists and overwrite is False."
            )

    sampler = sampler.lower()
    if sampler == "pymultinest":
        sampler = "multinest"

    # fmt: off
    if sampler == "ultranest":
        results = run_ultranest(post, output_dir=output_dir, sampler_kwargs=sampler_kwargs, run_kwargs=run_kwargs)
    elif sampler == "dynesty-static":
        results = run_dynesty(post, sampler_type="static", output_dir=output_dir, sampler_kwargs=sampler_kwargs, run_kwargs=run_kwargs)
    elif sampler == "dynesty-dynamic":
        results = run_dynesty(post, sampler_type="dynamic", output_dir=output_dir, sampler_kwargs=sampler_kwargs, run_kwargs=run_kwargs)
    elif sampler == "multinest":
        if sampler_kwargs is not None:
            raise TypeError("Argument sampler_kwargs is invalid for sampler 'multinest', only run_kwargs is supported")
        results = run_multinest(post, output_dir=output_dir, overwrite=overwrite, run_kwargs=run_kwargs)
    elif sampler == "nautilus":
        results = run_nautilus(post, output_dir=output_dir, sampler_kwarg=sampler_kwargs, run_kwargs=run_kwargs)
    else:
        raise ValueError(f"Unknown sampler '{sampler}'. Available options are {list(BACKENDS.keys())}")
    # fmt: on

    if output_dir is not None:
        with h5py.File(results_file, mode="w") as h5f:
            for key, val in results.items():
                if key == "sampler":
                    continue
                h5f.create_dataset(key, data=val)

    return results


def load_results(results_file: str) -> dict:
    """Load nested sampling results dictionary
    Args:
        results_file: Path to hdf5 file containing the results.
    Returns:
        Dictionary with nested sampling results.
        Note that the `sampler` key is not saved, so it is not in the dictionary returned by this function.
    """
    results = {}
    with h5py.File(results_file) as h5f:
        results["samples"] = np.array(h5f["samples"])
        results["lnZ"] = np.array(h5f["lnZ"]).item()
        results["lnZerr"] = np.array(h5f["lnZerr"]).item()
    return results
