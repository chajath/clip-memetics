# clip-memetics

CLIP memetics takes an evolutionary approach to generative art. That is, instead of optimizing a single lineage of GAN parameters, we populate a pool of candidate GAN parameters, and run a genetic algorithm to guide the population towards the text prompt goal.

Optimization goal is given as the mixture of image-text similarity and latent vector regularization.

This project is inspired by [CLIP-GLaSS project](https://github.com/galatolofederico/clip-glass). Our contribution is:

* The use of raw latent vector for BigGAN, rather than categorical binaries.
* Recording and rendering all results. This comes handy as a source material for secondary art creation.
* Interface to sample from previous runs. This allows users to intervene and change the evolution to the different direction by tweaking parameters and changing prompts.

## Setup

Follow the standard PyTorch setup for your platform. Then, install python dependencies:

```shell
$ pip install -r ./requirements.txt
```

## Synopsis

```shell
$ python main.py --prompt_text "What a wonderful world" --batch 5 --biggan_model=biggan-deep-128 --output_base ./data/wonderful1

$ python main.py --prompt_text "Trumpet sound" --batch 5 --biggan_model=biggan-deep-128 --output_base ./data/wonderful2 \
  --from_population="data/wonderful1/What a wonderful world-None-10-20.pt"
```

See help text from `python main.py --help` for the description of all available flags.
