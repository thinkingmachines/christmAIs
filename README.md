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

## It's christmAIs time!

We also provided a console interface to easily generate abstract art, `christmais_time.py`. 
Once you have installed `christmais` in your `site-packages`, then you can run the script via:

```shell
python -m christmais.scripts.christmais_time [--ARGS]
```

To see all available arguments, just pass `--help`. If you wish to set the colorscheme,
simply create a JSON file and pass it to the `--colorscheme` argument:

```json
# Sample color scheme for black lines only
# Background is white, circles are invisible, lines are black
{
    "background": [255, 255, 255, 255], 
    "layer1": [255, 255, 255, 0], 
    "layer2": [0, 0, 0, 0], 
    "layer3": [0, 0, 0, 0], 
    "lines": [0, 0, 0, 255]
}
```
