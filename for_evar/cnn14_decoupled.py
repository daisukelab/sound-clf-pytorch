"""CNN14 network, decoupled from Spectrogram, LogmelFilterBank, SpecAugmentation, and classifier head.

## Reference
- [1] https://arxiv.org/abs/1912.10211 "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition"
- [2] https://github.com/qiuqiangkong/audioset_tagging_cnn
"""

import torch
from torch import nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank


class AudioFeatureExtractor(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=512, n_mels=64, hop_length=160, win_length=512, f_min=50, f_max=8000):
        super().__init__()

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=n_fft, hop_length=hop_length, 
            win_length=win_length, window='hann', center=True, pad_mode='reflect', 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=win_length, 
            n_mels=n_mels, fmin=f_min, fmax=f_max, ref=1.0, amin=1e-10, top_db=None, 
            freeze_parameters=True)

    def forward(self, batch_audio):
        x = self.spectrogram_extractor(batch_audio) # (B, 1, T, F(freq_bins))
        x = self.logmel_extractor(x)                # (B, 1, T, F(mel_bins))
        return x


def initialize_layers(layer):
    # initialize all childrens first.
    for l in layer.children():
        initialize_layers(l)

    # initialize only linaer
    if type(layer) != nn.Linear:
        return

    # Thanks to https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4/pytorch/models.py#L10
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        initialize_layers(self.conv1)
        initialize_layers(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x


class Cnn14_Decoupled(nn.Module):
    """CNN14 network, decoupled from Spectrogram, LogmelFilterBank, SpecAugmentation, and classifier head.
    Original implementation: https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/master/pytorch/models.py
    """

    def __init__(self, n_mels=64, d=2048):
        assert d == 2048, 'This implementation accepts d=2048 only, for compatible with the original Cnn14.'
        super().__init__()

        self.bn0 = nn.BatchNorm2d(n_mels)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, d, bias=True)
        #self.fc_audioset = nn.Linear(d, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        initialize_layers(self.fc1)
        #init_layer(self.fc_audioset)

    def encode(self, x, squash_freq=True):
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x3 = x
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        if squash_freq:
            x = torch.mean(x, dim=3)
        return x

    def temporal_pooling(self, x):
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        return embedding

    def forward(self, x):
        x = self.encode(x)
        embedding = self.temporal_pooling(x)

        return embedding
