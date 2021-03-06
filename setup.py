"""Package setup file"""

from pathlib import Path

from setuptools import find_packages, setup

# Core package components and metadata

NAME = "genestboost"
AUTHOR = "Benjamin Cross"
EMAIL = "btcross26@yahoo.com"
PACKAGES = find_packages()
KEYWORDS = [""]
DESCRIPTION = "Generic estimator boosting"
LONG = "Generic boosting framework for any regression estimator"

PROJECT_URLS = {
    "Documentation": (
        "https://github.com/pages/btcross26/genestboost/build/html/index.html"
    ),
    "Bug Tracker": "https://github.com/btcross26/genestboost/issues",
    "Source Code": "https://github.com/btcross26/genestboost",
}

CLASSIFIERS = [
    "Intended Audience :: Everyone",
    "Natural Language :: English",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3.7",
]

EXTRAS_REQUIRE = {
    "docs": ["sphinx", "sphinx_rtd_theme", "recommonmark"],
    "tests": ["coverage", "mypy", "pytest", "pytest-cov", "toml"],
    "qa": [
        "pre-commit",
        "black",
        "mypy",
        "tox",
        "check-manifest",
        "flake8",
        "flake8-docstrings",
    ],
    "build": ["twine", "wheel"],
    "notebooks": [
        "jupyter",
        "ipykernel",
        "matplotlib==3.*,>=3.2.1",
        "pandas==1.*,>=1.0.4",
        "sklearn==0.*,>=0.22",
    ],
    "genestboost": ["genestboost"],
}

EXTRAS_REQUIRE["dev"] = (
    EXTRAS_REQUIRE["tests"]
    + EXTRAS_REQUIRE["docs"]
    + EXTRAS_REQUIRE["qa"]
    + EXTRAS_REQUIRE["build"]
    + EXTRAS_REQUIRE["notebooks"]
    + EXTRAS_REQUIRE["genestboost"]
)

HERE = Path(__file__).absolute().parent
INSTALL_REQUIRES = (HERE / "requirements.txt").read_text().split("\n")
VERSION = "0.1.0"
URL = PROJECT_URLS["Source Code"]
LICENSE = (HERE / "LICENSE").read_text()


def install_pkg():
    """Configure the setup for the package"""
    setup(
        name=NAME,
        version=VERSION,
        description=DESCRIPTION,
        long_description=LONG,
        url=URL,
        project_urls=PROJECT_URLS,
        author=AUTHOR,
        author_email=EMAIL,
        maintainer=AUTHOR,
        maintainer_email=EMAIL,
        license=LICENSE,
        python_requires=">=3.7.0",
        packages=PACKAGES,
        install_requires=INSTALL_REQUIRES,
        classifiers=CLASSIFIERS,
        extras_requires=EXTRAS_REQUIRE,
        include_package_data=True,
        zip_safe=False,
    )


if __name__ == "__main__":
    install_pkg()
