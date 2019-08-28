"""
Initial setup.py file for pip install -e
"""

from distutils.core import setup


setup(
    name="genestboost",
    version="0.1b",
    description="machine learning with general model boosting",
    author="Ben Cross",
    author_email="crossbt@vt.edu",
    py_modules=["genestboost",
                "genestboost.link_functions",
                "genestboost.loss_functions"],
)
