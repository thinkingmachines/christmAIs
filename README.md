# christmAIs

christmAIs ("krees-ma-ees") is text-to-abstract art generation for the holidays!

This project takes inspiration from Tom White's [perception
engines](https://medium.com/artists-and-machine-intelligence/perception-engines-8a46bc598d57)
and his [drawing system](https://github.com/dribnet/dopes) to generate abstract
art. 

Given a text input, a FastText model converts it into an 8-bit embedding, and
is used as a random seed for the drawing system. The generated images are then
fed to an ImageNet-trained classifier for prediction. The idea is that we keep
on perturbing the images until the classifier recognizes the target class
(tree, shopping cart, etc.).

## Requirements
- matplotlib==2.1.0
- setuptools==40.0.0
- gensim==3.5.0
- requests==2.18.4
- numpy==1.14.2
- nltk==3.2.4
- Pillow==5.3.0
- scikit_learn==0.20.0
- torch==0.4.1
- torchvision==0.2.1

## Set-up

First, clone this repository to your local machine:

```shell
$ git clone https://github.com/thinkingmachines/christmAIs.git
```

It is highly-recommended to use a virtual environment to set this project up
and install the dependencies:

```shell
$ cd christmAIs 
$ virtualenv venv
$ source venv/bin/activate
$ pip install -r requirements.txt # or requirements-dev.txt
```

## Usage

There are three important components for this perception engine to work:
- `christmais.FastTextWrapper`: maps a string into an 8-bit vector
- `christmais.Artist`: maps an 8-bit vector into an image
- `christmais.Predictor`: classifies an image into a particular object

In addition, there is also a `christmais.Trainer` class that takes all three
components, then performs a random walk in order to find the abstract art that
best resembles a target object.

### Map a string into an 8-bit vector

This module contains a wrapper for `gensim.FastText` to create word embeddings
for a given text.

```python
from christmais import FastTextWrapper
from nltk.corpus import brown # or any other corpus

# Train the model
model = FastTextWrapper(sentences=brown.sents())
# Embed a text
my_text = "Thinking Machines Data Science"
model.transform(my_text)
```

Or, you can simply use a pre-trained FastText model on the brown corpus (note:
if no `.model` is found in your `/tmp/` directory, then it trains as usual):

```python
from christmais import get_fasttext_pretrained
# Assuming that /tmp/brown_fasttext.model exists
model = get_fasttext_pretrained(load=True)
# Embed a text
my_text = "Thinking Machines Data Science"
model.transform(my_text)
```

### Map an 8-bit vector into an image

Once you have generated a word embedding, you can then use it as a seed to the
drawing system, a.k.a. the `Artist` class:

```python
from christmais import (get_fasttext_pretrained, Artist)

model = get_fasttext_pretrained(load=True)
seed = model.transform("Thinking Machines Data Science")
artist = Artist(seed, dims=(224, 224)) 
artist.draw()
```
