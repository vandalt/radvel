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

    if sys.version_info[0] < 3:
        ep = ExecutePreprocessor(timeout=900, kernel_name="python2")
    else:
        ep = ExecutePreprocessor(timeout=900, kernel_name="python3")

    ep.preprocess(nb, {"metadata": {"path": os.path.dirname(nbfile)}})
