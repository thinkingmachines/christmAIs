============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given. 

.. toctree::
   :numbered:
   :maxdepth: 2

   types_of_contributions
   contributor_guidelines

Types of Contributions
----------------------

There are many ways to contribute in this project:

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/thinkingmachines/christmAIs/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* If you can, provide detailed steps to reproduce the bug.
* If you don't have steps to reproduce the bug, just note your observations in
  as much detail as you can. Questions to start a discussion about the issue
  are welcome.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug"
is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "please-help" is open to whoever wants to implement it.

Please do not combine multiple feature enhancements into a single pull request.

Note: this project is very conservative, so new features that aren't tagged
with "please-help" might not get into core. We're trying to keep the code base
small, extensible, and streamlined. Whenever possible, it's best to try and
implement feature ideas as separate projects outside of the core codebase.

Write Documentation
~~~~~~~~~~~~~~~~~~~

christmAIs could always use more documentation, whether as part of the
official Cookiecutter docs, in docstrings, or even on the web in blog posts,
articles, and such.

If you want to review your changes on the documentation locally, you can do::

    pip install -r requirements-dev.txt
    cd docs
    make html

This will compile the documentation, open it in your browser and start
watching the files for changes, recompiling as you save.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at
https://github.com/thinkingmachines/christmAIs/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
are welcome :)


Contributor Guidelines
----------------------

Pull Request Guidelines
~~~~~~~~~~~~~~~~~~~~~~~

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.rst.
3. The pull request should work for Python 3.6

Coding Standards
~~~~~~~~~~~~~~~~

* We use PEP8 as our coding standard
* In addition, we use `black <https://github.com/ambv/black>`_
