import warnings
import os
import sys
from glob import glob

import pytest
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

warnings.simplefilter("ignore")


@pytest.mark.parametrize(
    "nbfile", sorted(glob(os.path.join("docs/tutorials/", "*.ipynb")))
)
def test_notebook(nbfile):
    """
    Run though a single notebook tutorial
    """
    print(nbfile)
    with open(nbfile) as f:
        nb = nbformat.read(f, as_version=4)

    basename = os.path.basename(nbfile)
    # Skip the all samplers notebook. Same functionality tested in api tests and it is slow due to repeated sampling to convergence.
    skip_notebooks = ["k2_24_demo_all_samplers.ipynb"]
    if basename in skip_notebooks:
        return
    timeout = 900

    if sys.version_info[0] < 3:
        ep = ExecutePreprocessor(timeout=timeout, kernel_name="python2")
    else:
        ep = ExecutePreprocessor(timeout=timeout, kernel_name="python3")

    ep.preprocess(nb, {"metadata": {"path": os.path.dirname(nbfile)}})
