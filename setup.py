import setuptools
from numpy.distutils.core import Extension, setup
from l1periodogram.version import PACKAGE_NAME, PACKAGE_VERSION, PACKAGE_AUTHOR


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

gglasso = Extension("l1periodogram.gglasso_wrapper", ["l1periodogram/gglasso.f90"])

setup(
    name=PACKAGE_NAME,
    version=PACKAGE_VERSION,
    author=PACKAGE_AUTHOR,
    author_email="nathan.hara@unige.ch",
    description="A small Python package to search for periodicities in unevenly sampled time series.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nathanchara/l1periodogram",
    project_urls={
        "Bug Tracker": "https://github.com/nathanchara/l1periodogram/issues",
    },
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    ext_modules = [gglasso],
    python_requires='>=3.6',
    install_requires=[
        "pandas",
        "numpy",
        "scipy",
        "matplotlib",
    ]
)
