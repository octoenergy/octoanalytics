from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.2.0'
DESCRIPTION = 'Quantitative analysis for power markets'
LONG_DESCRIPTION = 'An internal Python package offering tools for quantitative & risk analysis on power markets.'


# Setting up
setup(
    name="octoanalytics",
    version=VERSION,
    author="Jean Bertin",
    author_email="<jean.bertin@octopusenergy.fr>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "holidays"
    ],
    keywords=['python', 'quantitative', 'risk', 'power', 'quant','octopusenergy'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
