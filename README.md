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

### Installing magenta

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

### Downloading the pretrained model

A pretrained model based from the Painter by Number (PBN) dataset and Describable
Textures Dataset (DTD) is provided (around 600+ MB). To download them, simply
execute the following command:

```shell
$ mkdir ckpt && \
  wget https://storage.googleapis.com/download.magenta.tensorflow.org/models/arbitrary_style_transfer.tar.gz && \
  tar --strip-components 1 -xvzf arbitrary_style_transfer.tar.gz -C ckpt/   
```

### Installing everything else

You can then install the remaining dependencies in `requirements.txt`. Assuming
that you have create a virtual environment via `make venv`, you can simply run
the following command:

```shell
$ make build # or `make dev`
```
