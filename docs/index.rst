==========
christmAIs
==========

.. image:: https://storage.googleapis.com/tm-github-builds/build/christmAIs-master.svg
   :target: https://console.cloud.google.com/cloud-build/builds?authuser=2&organizationId=301224238109&projec|t=tm-github-builds
   :alt: Build Status
.. image:: https://readthedocs.org/projects/christmais-2018/badge/?version=latest
   :target: https://christmais-2018.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
.. image:: https://img.shields.io/badge/License-GPL%20v2-blue.svg
   :target: https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html
   :alt: License
.. image:: https://img.shields.io/badge/python-3.6+-blue.svg
   :alt: Python version

**christmAIs** ("krees-ma-ees") is text-to-abstract art generation for the
holidays!

This work converts any input string into an abstract art by:

* finding the most similar `Quick, Draw! <https://quickdraw.withgoogle.com/data>`_ class using `GloVe <https://nlp.stanford.edu/projects/glove/>`_
* drawing the nearest class using a Variational Autoencoder (VAE) called `Sketch-RNN <https://arxiv.org/abs/1704.03477>`_; and
* applying `neural style transfer <https://arxiv.org/abs/1508.06576>`_ to the resulting image

This results to images that look like these:

.. image:: https://storage.googleapis.com/tm-christmais/assets/book1.png
   :height: 170
.. image:: https://storage.googleapis.com/tm-christmais/assets/book2.png
   :height: 170
.. image:: https://storage.googleapis.com/tm-christmais/assets/sf1.png
   :height: 170
.. image:: https://storage.googleapis.com/tm-christmais/assets/truck1.png
   :height: 170

Table of Contents
------------------

.. toctree::
   :maxdepth: 4

   setup
   usage
   christmais
   contributing
   references
   


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
