# christmAIs

**christmAIs** ("krees-ma-ees") is text-to-abstract art generation for the holidays!

This work converts any input string into an abstract art by:
- finding the most similar [Quick, Draw!](https://quickdraw.withgoogle.com/data) class using [GloVe](https://nlp.stanford.edu/projects/glove/)
- drawing the nearest class using a Variational Autoencoder (VAE) called [Sketch-RNN](https://arxiv.org/abs/1704.03477); and
- applying [neural style transfer](https://arxiv.org/abs/1508.06576) to the resulting image

This results to images that look like this:

![alt text](https://raw.githubusercontent.com/thinkingmachines/christmAIs/master/assets/book1.png?token=AMWYs2z_JoFRncHWEjer7NP_aUQ20G2pks5cDc8gwA%3D%3D)
![alt text](https://raw.githubusercontent.com/thinkingmachines/christmAIs/master/assets/book2.png?token=AMWYswJpjY4WYEoOeQxy84ziXDFj1ueaks5cDc9dwA%3D%3D)
![alt text](https://raw.githubusercontent.com/thinkingmachines/christmAIs/master/assets/sf1.png?token=AMWYsxAr2m8Nc7UiermGFKgd9Z6atjuLks5cDc9fwA%3D%3D)
![alt text](https://raw.githubusercontent.com/thinkingmachines/christmAIs/master/assets/truck1.png?token=AMWYs2dz3AMOGdS1ScaCGBWyvo-_VxRgks5cDdBvwA%3D%3D)

## Setup

Please see `requirements.txt` and `requirements-dev.txt`. In addition, see
`build.sh` to see the setup steps needed for building your environment. 

In addition, we provided a `Makefile` to ease things up:

```shell
$ git clone git@github.com:thinkingmachines/christmAIs.git
$ cd christmaAIs
$ make venv
$ make build # or make dev
```

