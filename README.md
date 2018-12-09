# christmAIs

[![License: GPL v2](https://img.shields.io/badge/License-GPL%20v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)
![python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)

**christmAIs** ("krees-ma-ees") is text-to-abstract art generation for the
holidays!

This work converts any input string into an abstract art by:
- finding the most similar [Quick, Draw!](https://quickdraw.withgoogle.com/data) class using [GloVe](https://nlp.stanford.edu/projects/glove/)
- drawing the nearest class using a Variational Autoencoder (VAE) called [Sketch-RNN](https://arxiv.org/abs/1704.03477); and
- applying [neural style transfer](https://arxiv.org/abs/1508.06576) to the resulting image

This results to images that look like these:

![alt text](https://raw.githubusercontent.com/thinkingmachines/christmAIs/master/assets/book1.png?token=AMWYs2z_JoFRncHWEjer7NP_aUQ20G2pks5cDc8gwA%3D%3D)
![alt text](https://raw.githubusercontent.com/thinkingmachines/christmAIs/master/assets/book2.png?token=AMWYswJpjY4WYEoOeQxy84ziXDFj1ueaks5cDc9dwA%3D%3D)
![alt text](https://raw.githubusercontent.com/thinkingmachines/christmAIs/master/assets/sf1.png?token=AMWYsxAr2m8Nc7UiermGFKgd9Z6atjuLks5cDc9fwA%3D%3D)
![alt text](https://raw.githubusercontent.com/thinkingmachines/christmAIs/master/assets/truck1.png?token=AMWYs2dz3AMOGdS1ScaCGBWyvo-_VxRgks5cDdBvwA%3D%3D)

## Setup and Installation

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
[Dockerfile](https://github.com/thinkingmachines/christmAIs/blob/master/Dockerfile).
For local development, it is **recommended to setup a virtual environment**. To
do that, simply run the following commands:

```shell
$ git clone git@github.com:thinkingmachines/christmAIs.git
$ cd christmAIs
$ make venv
```

### Automated Install

We created an automated install script to perform a one-click setup in your
workspace. To run the script, execute the following command:

```shell
$ curl https://storage.googleapis.com/tm-christmais/install-christmais.sh > install-christmais.sh
$ bash install-christmais.sh
```

It is recommended to perform these steps inside a virtual environment. It will
first install `magenta` and its dependencies, download file dependencies
(`categories.txt`, `model.ckpt`, and `chromedriver`), then clone and install
this package.

### Manual Install

For manual installation, please follow the instructions below:

#### Installing magenta

The style transfer capabilities are dependent on the
[magenta](https://github.com/tensorflow/magenta) package. As of now, magenta is
only supported in Linux and Mac OS. To install magenta, you can perform the
[automated install](https://github.com/tensorflow/magenta#automated-install)
or do the following steps:

```shell
# Install OS dependencies 
$ apt-get update && \
  apt-get install -y build-essential libasound2-dev libjack-dev

# Install magenta
$ venv/bin/pip install magenta

```

#### Installing everything else

You can then install the remaining dependencies in `requirements.txt`. Assuming
that you have create a virtual environment via `make venv`, we recommend that
you simply run the following command:

```shell
$ make build # or `make dev`
```

This will also download (via `wget`) the following files:
- **categories.txt** (683 B): contains the list of Quick, Draw! categories to compare a string upon (will be saved at `./categories/categories.txt`).
- **arbitrary_style_transfer.tar.gz** (606.20 MB): contains the model checkpoint for style
    transfer (will be saved at `./ckpt/model.ckpt`).
- **chromedriver** (5.09 MB): contains the web driver for accessing the HTML output for
    Sketch-RNN (will be saved at `./webdriver/chromedriver`).

## Usage

We have provided a script, `christmais_time.py` to easily generate your stylized Quick, Draw! images.
In order to use it, simply run the following command:

```shell
$ python -m christmais.tasks.christmais_time   \
    --input=<Input string to draw from>        \
    --style=<Path to style image>              \
    --output=<Unique name of output file>      \
    --model-path=<Path to model.ckpt>          \
    --categories-path=<Path to categories.txt> \
    --webdriver-path=<Path to webdriver> 
```

If you followed the setup instructions above, then the default values for the
paths should suffice, you only need to supply `--input`, `--style`, and
`--output`.

As an example, let's say I want to use the string `Thinking Machines` as our
basis with the style of *Ang Kiukok* (`ang_kiukok.jpg`), then, my command will
look like this:

```shell
$ python -m christmais.tasks.christmais_time    \
    --input="Thinking Machines"                 \
    --style=./path/to/ang_kiukok.png            \
    --output=tmds-output                         
```

This will then generate the output image in `./artifacts/`:

![alt text](https://raw.githubusercontent.com/thinkingmachines/christmAIs/master/assets/tmds.png?token=AMWYs2dz3AMOGdS1ScaCGBWyvo-_VxRgks5cDdBvwA%3D%3D)
