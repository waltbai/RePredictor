"""Setup for repredictor."""

from setuptools import setup, find_packages


def _process_requirements():
    with open("requirements.txt", "r") as f:
        packages = f.read().strip().splitlines()
    return packages


setup(
    name="repredictor",
    version="0.1",
    author="Long Bai",
    author_email="bailong18b@ict.ac.cn",
    description="Rich Event Representation for Script Event Prediction",
    # Requirements
    packages=find_packages(),
    install_requires=_process_requirements(),
)
