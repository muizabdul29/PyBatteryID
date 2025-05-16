"""Setup for the PyBatteryID package"""

# pylint: disable=E0401
from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    readme = f.read()

setup(
    name='pybatteryid',
    version='2.0.1',
    author='Muiz Sheikh',
    description='Data-driven Battery Model Identification in Python',
    long_description=readme,
    long_description_content_type='text/markdown',
    url="https://github.com/muizabdul29/PyBatteryID",
    packages=find_packages(include=['pybatteryid', 'pybatteryid.*']),
    install_requires=[
        'numpy>=1.20.0',
        'cvxopt',
        'scikit-learn',
        'rich',
        'matplotlib'
    ],
    tests_require=[],
    classifiers = [
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
