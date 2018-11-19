# -*- coding: utf-8 -*-

# Import modules
from setuptools import find_packages, setup

with open("README.md", encoding="utf8") as readme_file:
    readme = readme_file.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open('requirements-dev.txt') as f:
    dev_requirements = f.read().splitlines()

setup(
    name="christmais",
    version="1.0.0",
    description="Text to abstract art for the holidays!",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Thinking Machines Data Science",
    author_email="hello@thinkingmachin.es",
    url="https://github.com/thinkingmachines/christmAIs",
    packages=find_packages(exclude=["docs", "tests"]),
    include_package_data=True,
    install_requires=requirements,
    tests_require=dev_requirements,
    extras_require={"test": dev_requirements},
    license="MIT license",
    zip_safe=False,
    keywords="christmas",
    classifiers=[
        "Development Status :: 1 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
    ],
    test_suite="tests",
)
