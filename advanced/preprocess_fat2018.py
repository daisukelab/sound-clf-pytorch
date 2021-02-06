"""Preprocess Freesound Audio Tagging 2018 competition data.
"""

import warnings
warnings.simplefilter('ignore')

from src.libs import *
from tqdm import tqdm
import fire

def convert(config='config.yaml'):
    cfg = load_config(config)
    print(cfg)
    DATA_ROOT = Path(cfg.data_root)
    DEST = Path('work')/cfg.type

    folders = ['FSDKaggle2018.audio_test', 'FSDKaggle2018.audio_train']

    to_mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg.sample_rate, n_fft=cfg.n_fft, n_mels=cfg.n_mels,
        hop_length=cfg.hop_length, f_min=cfg.f_min, f_max=cfg.f_max)

    for folder in folders:
        cur_folder = DATA_ROOT/folder
        filenames = sorted(cur_folder.glob('*.wav'))
        resampler = None
        for filename in tqdm(filenames):
            # Load waveform
            waveform, sr = torchaudio.load(filename)
            #assert sr == cfg.sample_rate
            if sr != cfg.sample_rate:
                if resampler is None:
                    resampler = torchaudio.transforms.Resample(sr, cfg.sample_rate)
                    print(f'CAUTION: RESAMPLING from {sr} Hz to {cfg.sample_rate} Hz.')
                waveform = resampler(waveform)
            # To log-mel spectrogram
            log_mel_spec = to_mel_spectrogram(waveform).log()
            # Write to work
            (DEST/folder).mkdir(parents=True, exist_ok=True)
            np.save(DEST/folder/filename.name.replace('.wav', '.npy'), log_mel_spec)


fire.Fire(convert)
