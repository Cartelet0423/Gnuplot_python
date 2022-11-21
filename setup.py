"""Minimal setup file for tasks project."""

from setuptools import setup, find_packages

setup(
    name="Gnuplot",
    version="0.1.0",
    license="proprietary",
    description="The Easy-to-use PyGnuplot Wrapper.",
    author="Cartelet0423",
    author_email="",
    url="",
    packages=find_packages(),
    install_requires=["numpy", "PyGnuplot"],
    python_requires=">=3.6",
)
