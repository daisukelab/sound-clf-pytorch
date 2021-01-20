# Sound Classifier Tutorial for PyTorch

This is sound classifier tutorials using PyTorch, PyTorch Lightning and torchaudio.

## 0. Motivation

I had made a repository regarding sound classifier solution: [Machine Learning Sound Classifier for Live Audio](https://github.com/daisukelab/ml-sound-classifier),
it is based on my solution for a Kaggle competition "[Freesound General-Purpose Audio Tagging Challenge](https://www.kaggle.com/c/freesound-audio-tagging)" using Keras.

Keras was popular when it was created, but many people today are using PyTorch, and yes, I'm also the one.

This repository is an updated example solution using PyTorch that shows how I would try new machine learning sound competition with current software assets.

## 1. Quickstart

- `pip install -r requirements.txt` to install modules.
- Run notebooks.

## 2. What you can find

### 2-1. What's included

- An audio preprocessing example: [Data-Preprocessing.ipynb](Data-Preprocessing.ipynb)
- A training example: [Training-Classifier.ipynb](Training-Classifier.ipynb)
- [FSDKaggle2018](https://zenodo.org/record/2552860#.X9TH6mT7RzU) handling example, it's a sound multi-class classification task.
- New) ResNetish/VGGish [1] models.
    - Models are equipped with AdaptiveXXXPool2d to be flexible with input size. Models now accept any shape.
- New) Colab all-in-one notebook [Run-All-on-Colab.ipynb](Run-All-on-Colab.ipynb). You can run all through the training/evaluation online.

### 2-2. What's not

- No usual practices/techniques: Normalization, augmentation, regularizations and etc. --> will be followed up with advanced notebook.
- No cutting edge networks like what you can find there: [PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition](https://github.com/qiuqiangkong/audioset_tagging_cnn).

You can just run to reproduce, or try advanced techniques based on the tutorials.

## 3. Notes for design choices

### 3-1. Input data format: raw audio or spectrogram?

If we need to augment input data in time-domain, we feed raw audio to dataset class.

But in this example, all the data are converted to log-mel spectrogram in advance, as a major choice.

- Good: This will make data handling easy, especially in training pipeline.
- Bad: Applicable data augmentations will be limited. Available transformations in torchaudio are: [FrequencyMasking](https://pytorch.org/audio/stable/transforms.html#frequencymasking) or [TimeMasking](https://pytorch.org/audio/stable/transforms.html#timemasking).

### 3-2. Input data size

Number of frequency bins (n_mels) is set to 64 as a typical choice.
Duration is set to ~~1 second, just as an example.~~ 5 seconds in current configuration, because 1 second was too short for the FSDKaggle2018 dataset.

You can find and change in [config.yaml](config.yaml).

    clip_length: 5.0 # [sec] -- it was 1.0 s at the initial release.
    n_mels: 64

### 3-3. FFT paramaters

Typical paramaters are configured in [config.yaml](config.yaml).

    sample_rate: 44100
    hop_length: 441
    n_fft: 1024
    n_mels: 64
    f_min: 0
    f_max: 22050

## 4. Performances

How is the performance of the trained models on the tutorials?

- The best Kaggle result MAP@3 was reported as 0.942 (see [Kaggle 4th solution](https://www.kaggle.com/c/freesound-audio-tagging/discussion/62634)). Note that this result is ensemble of 5 models of the same SE-ResNeXt network trained on 5 folds.
- The best result in this repo is MAP@3 of 0.87 (with ResNetish). This is a single model result, without use of data augmentations.

Already came close to the top solution with ResNetish, and still have space for data augmentations/ regularization techniques.

## References

- [1] S. Hershey et al., ‘CNN Architectures for Large-Scale Audio Classification’,\ in International Conference on Acoustics, Speech and Signal Processing (ICASSP),2017\ Available: https://arxiv.org/abs/1609.09430, https://ai.google/research/pubs/pub45611
