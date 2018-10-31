# -*- coding: utf-8 -*-

# Import modules
from setuptools import find_packages, setup

with open("README.md", encoding="utf8") as readme_file:
    readme = readme_file.read()

requirements = [
    "nltk==3.2.4",
    "gensim==3.5.0",
    "numpy==1.14.2",
    "Pillow==5.3.0",
    "scikit_learn==0.20.0",
]

test_requirements = [
    "pytest==3.6.4",
    "tox==3.2.1",
    "flake8==3.5.0",
    "python-dotenv==0.9.1",
]

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
    tests_require=test_requirements,
    extras_require={"test": test_requirements},
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
