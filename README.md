# Sound Classifier Tutorial for PyTorch

This is a sound classifier tutorial based on PyTorch, PyTorch Lightning and torchaudio.

## Motivation

I had made a repository for sound classifier [Machine Learning Sound Classifier for Live Audio](https://github.com/daisukelab/ml-sound-classifier),
it is based on my solution for the Kaggle competition "[Freesound General-Purpose Audio Tagging Challenge](https://www.kaggle.com/c/freesound-audio-tagging)" with Keras.

Now many people are using PyTorch, and yes, I'm also the one.

This repository shows a quick example for how I would try new sound machine learning competition with current software assets.

## Quickstart

- `pip install -r requirements.txt` to install modules.
- Run notebooks.

## What's included

- An audio preprocessing example: [Data-Preprocessing.ipynb](Data-Preprocessing.ipynb)
- A training example: [Training-Classifier.ipynb](Training-Classifier.ipynb)
- [FSDKaggle2018](https://zenodo.org/record/2552860#.X9TH6mT7RzU) handling example, it's a sound multi-class classification task.

It's about accuracy ~0.7, far away from the competitive accuracy ~0.95 you can find here: [Kaggle top solutions](https://www.kaggle.com/c/freesound-audio-tagging/leaderboard).

## What's not

- Some usual practices for getting higher accuracy: Normalization, augmentation, regularizations and etc.
- Better network like what you can find there: [PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition](https://github.com/qiuqiangkong/audioset_tagging_cnn).

You can try many techniques based on this simple tutorial, so I'd keep things simple here.

## What can happen in the future

Following examples for getting better results.

