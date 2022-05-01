"""webdataset
"""

import sys
from multiprocessing import Pool
from pathlib import Path
import pandas as pd
import webdataset as wds
from itertools import islice
import librosa
import fire


def fsd50k_metadata(FSD50K_root):
    FSD = Path(FSD50K_root)
    df = pd.read_csv(FSD/f'FSD50K.ground_truth/dev.csv')
    df['key'] = df.split + '_' + df.fname.apply(lambda s: str(s))
    df['fname'] = df.fname.apply(lambda s: f'FSD50K.dev_audio/{s}.wav')
    dftest = pd.read_csv(FSD/f'FSD50K.ground_truth/eval.csv')
    dftest['key'] = 'eval_' + dftest.fname.apply(lambda s: str(s))
    dftest['split'] = 'eval'
    dftest['fname'] = dftest.fname.apply(lambda s: f'FSD50K.eval_audio/{s}.wav')
    df = pd.concat([df, dftest], ignore_index=True)
    return df


def load_resampled_mono_wav(fpath, sr):
    y, org_sr = librosa.load(fpath, sr=None, mono=True)
    if org_sr != sr:
        y = librosa.resample(y, orig_sr=org_sr, target_sr=sr)
    return y


def _converter_worker(args):
    fpath, sr = args
    return load_resampled_mono_wav(fpath, sr)


def fsd50k_generator(root, split, sr):
    root = Path(root)
    df = fsd50k_metadata(FSD50K_root=root)
    df = df[df.split == split]
    print(f'Processing {len(df)} {split} samples.')
    for file_name, labels, key in df[['fname', 'labels', 'key']].values:
        fpath = root/file_name

        sample = {
            '__key__': key,
            'npy': fpath, # load_resampled_mono_wav(fpath, sr),
            'labels': labels,
        }
        yield sample


def create_wds(source, output, split, sr, name='fsd50k-[SPLIT]-[SR]-%06d.tar', maxsize=10**9):
    source = source
    name = name.replace('[SPLIT]', split).replace('[SR]', str(sr))
    output_name = str(Path(output)/name)

    gen = fsd50k_generator(source, split, sr)
    with wds.ShardWriter(output_name, maxsize=maxsize) as sink:
        while True:
            samples = list(islice(gen, 100))
            if len(samples) == 0:
                break
            # load and resample wav files
            with Pool() as p:
                args = [[s['npy'], sr] for s in samples]
                npys = list(p.imap(_converter_worker, args))
                for s, npy in zip(samples, npys):
                    s['npy'] = npy
                    sink.write(s)
            print('.', end='')
            sys.stdout.flush()
    print('Finished')


if __name__ == '__main__':
    fire.Fire(create_wds)
