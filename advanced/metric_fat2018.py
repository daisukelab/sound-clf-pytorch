# Based on https://github.com/DCASE-REPO/dcase2018_baseline/blob/master/task2/evaluation.py

from src.libs import *
import datetime
import numpy as np
import torch
import multiprocessing


def one_ap(gt, topk):
    for i, p in enumerate(topk):
            if gt == p:
                return 1.0 / (i + 1.0)
    return 0.0


def avg_precision(gts=None, topks=None):
    return np.array([one_ap(gt, topk) for gt, topk in zip(gts, topks)])


def eval_fat2018_by_probas(probas, labels, debug_name=None, TOP_K=3):
    correct = ap = 0.0
    for proba, label in zip(probas, labels):
        topk = proba.argsort()[-TOP_K:][::-1]
        correct += int(topk[0] == label)
        ap += one_ap(label, topk)
    acc = correct / len(labels)
    mAP = ap / len(labels)
    if debug_name:
        print(f'{debug_name} acc = {acc:.4f}, MAP@{TOP_K} = {mAP}')
    return acc, mAP


def eval_fat2018(model, device, dataloader, debug_name=None, TTA=1):
    model = model.to(device).eval()
    all_probas, labels = [], []
    with torch.no_grad():
        for _ in range(TTA):
            for X, gts in dataloader:
                preds = model(X.to(device))
                probas = preds.softmax(1)
                all_probas.extend(probas.cpu().numpy())
                labels.extend(gts.cpu().numpy())
    all_probas = np.array(all_probas)
    return eval_fat2018_by_probas(all_probas, labels, debug_name=debug_name), all_probas


def eval_fat2018_all_splits(cfg, model, device, filenames, labels, norm_mean_std=None, debug_name=None, head_n=999, agg='mean'):
    model = model.to(device).eval()
    file_probas = [[] for _ in range(len(labels))]
    test_dataset = SplitAllDataset(cfg, filenames, norm_mean_std=norm_mean_std, head_n=head_n)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.bs,
                                              num_workers=multiprocessing.cpu_count(), pin_memory=True)
    print(f'Predicting all {len(test_dataset)} splits for {len(labels)} files...')
    for X, fileidxs in test_loader:
        with torch.no_grad():
            preds = model(X.to(device))
            probas = F.softmax(preds, dim=1)
        for idx, prob in zip(fileidxs.cpu().numpy(), probas.cpu().numpy()):
            file_probas[idx].append(prob)

    if agg == 'max':
        file_probas = np.array([np.max(probas, axis=0) for probas in file_probas])
    elif agg == 'mean':
        file_probas = np.array([np.mean(probas, axis=0) for probas in file_probas])
    else:
        raise Exception()

    return eval_fat2018_by_probas(file_probas, labels, debug_name=debug_name), file_probas
