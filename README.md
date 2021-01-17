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
- New) ResNetish/VGGish [1] models.
    - Models are equipped with AdaptiveXXXPool2d to be flexible with input size. Models now accept any shape.

## What's not

- Simple example, but not enough performance to compete in Kaggle.
    - Accuracy was ~0.7 at rge initial release.
    - New) The latest accuracy is ~0.83 with ResNetish.
    - Still far away from the cutting edge accuracy ~0.95 which you can find in [Kaggle top solutions](https://www.kaggle.com/c/freesound-audio-tagging/leaderboard).
- Some usual practices for getting higher accuracy: Normalization, augmentation, regularizations and etc.
- Better network like what you can find there: [PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition](https://github.com/qiuqiangkong/audioset_tagging_cnn).

You can try many techniques based on this simple tutorial, so I'd keep things simple here.

## Notes for design choices

### Input data format: raw audio or spectrogram?

If we need to augment input data in time-domain, we feed raw audio to dataset class.

But in this example, all the data are converted to log-mel spectrogram in advance, as a major choice.

- Good: This will make data handling easy, especially in training pipeline.
- Bad: Data augmentation will be limited. Available transforations in torchaudio are: [FrequencyMasking](https://pytorch.org/audio/stable/transforms.html#frequencymasking) or [TimeMasking](https://pytorch.org/audio/stable/transforms.html#timemasking).

### Input data size

Number of frequency bins (n_mels) is set to 64 as a typical choice.
Duration is set to ~~1 second, just as an example.~~ 5 seconds in current configuration, because 1 second was too short for the FSDKaggle2018 dataset.

You can find and change in [config.yaml](config.yaml).

    clip_length: 5.0 # [sec] -- it was 1.0 s at the initial release.
    n_mels: 64

### FFT paramaters

Typical paramaters are configured in [config.yaml](config.yaml).

    sample_rate: 44100
    hop_length: 441
    n_fft: 1024
    n_mels: 64
    f_min: 0
    f_max: 22050

## References

- [1] S. Hershey et al., ‘CNN Architectures for Large-Scale Audio Classification’,\ in International Conference on Acoustics, Speech and Signal Processing (ICASSP),2017\ Available: https://arxiv.org/abs/1609.09430, https://ai.google/research/pubs/pub45611
