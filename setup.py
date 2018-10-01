from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="LDM-Query",
    version="0.1",
    author="Cai Wingfield",
    author_email="c.wingfield@lancaster.ac.uk",
    description="Query linguistic distributional models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/emcoglab/ldm-query",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
