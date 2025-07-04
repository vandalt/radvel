from typing import Optional

import ultranest

from radvel.posterior import Posterior


def nested_sampling(
    post: Posterior,
    sampler_kwargs: Optional[dict] = None,
    run_kwargs: Optional[dict] = None,
) -> ultranest.ReactiveNestedSampler:
    run_kwargs = run_kwargs or {}
    sampler_kwargs = sampler_kwargs or {}
    post.check_proper_priors()
    sampler = ultranest.ReactiveNestedSampler(
        post.name_vary_params(),
        post.likelihood_ns_array,
        post.prior_transform,
        **sampler_kwargs,
    )

    sampler.run(**run_kwargs)

    return sampler
