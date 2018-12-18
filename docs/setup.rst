======================
Setup and Installation
======================

Please see `requirements.txt` and `requirements-dev.txt` for all Python-related
dependencies. Notable dependencies include:

- numpy==1.14.2
- scikit_learn==0.20.0
- Pillow==5.3.0
- matplotlib==2.1.0
- tensorflow
- gensim
- magenta

The build steps (what we're using to do automated builds in the cloud) can be
seen in the
`Dockerfile <https://github.com/thinkingmachines/christmAIs/blob/master/Dockerfile>`_.
For local development, it is **recommended to setup a virtual environment**. To
do that, simply run the following commands:

.. code-block:: shell

   git clone git@github.com:thinkingmachines/christmAIs.git
   cd christmAIs
   make venv

Automated Install
-----------------

We created an automated install script to perform a one-click setup in your
workspace. To run the script, execute the following command:

.. code-block:: shell

   source venv/bin/activate  # Highly recommended
   ./install-christmais.sh

This will first install `magenta` and its dependencies, download file
dependencies (`categories.txt`, `model.ckpt`, and `chromedriver`), then clone
and install this package.

Manual Install
--------------

For manual installation, please follow the instructions below:

Installing magenta
~~~~~~~~~~~~~~~~~~

The style transfer capabilities are dependent on the
`magenta <https://github.com/tensorflow/magenta>`_ package. As of now, magenta is
only supported in Linux and Mac OS. To install magenta, you can perform the
`automated install <https://github.com/tensorflow/magenta#automated-install>`_
or do the following steps:

.. code-block:: shell

   # Install OS dependencies
   apt-get update && \
   apt-get install -y build-essential libasound2-dev libjack-dev

   # Install magenta
   venv/bin/pip install magenta


Installing everything else
~~~~~~~~~~~~~~~~~~~~~~~~~~

You can then install the remaining dependencies in `requirements.txt`. Assuming
that you have create a virtual environment via `make venv`, we recommend that
you simply run the following command:

.. code-block:: shell

   make build # or make dev

This will also download (via `wget`) the following files:

* **categories.txt** (683 B): contains the list of Quick, Draw! categories to compare a string upon (will be saved at `./categories/categories.txt`).
* **arbitrary_style_transfer.tar.gz** (606.20 MB): contains the model checkpoint for style transfer (will be saved at `./ckpt/model.ckpt`).
* **chromedriver** (5.09 MB): contains the web driver for accessing the HTML output for Sketch-RNN (will be saved at `./webdriver/chromedriver`).
