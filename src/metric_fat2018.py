# https://github.com/DCASE-REPO/dcase2018_baseline/blob/master/task2/evaluation.py
import numpy as np
import torch


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
