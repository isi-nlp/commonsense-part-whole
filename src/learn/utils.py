import json
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_squared_error, precision_score, recall_score)
import torch

def tuple_collate(batch):
    return [[b[i] for b in batch] for i in range(len(batch[0]))]

def dfn_collate(batch):
    wdfns, pdfns, jdfns, labels = [], [], [], []
    wlen, plen, jlen = 0, 0, 0
    for (wdfn, pdfn, jdfn), label in batch:
        wdfns.append(wdfn.split())
        pdfns.append(pdfn.split())
        jdfns.append(jdfn.split())
        labels.append(label)
        if len(wdfn.split()) > wlen:
            wlen = len(wdfn.split())
        if len(pdfn.split()) > plen:
            plen = len(pdfn.split())
        if len(jdfn.split()) > jlen:
            jlen = len(jdfn.split())
    for wdfn in wdfns:
        if len(wdfn) < wlen:
            wdfn.extend(["*PAD*"] * (wlen - len(wdfn)))
    for pdfn in pdfns:
        if len(pdfn) < plen:
            pdfn.extend(["*PAD*"] * (plen - len(pdfn)))
    for jdfn in jdfns:
        if len(jdfn) < jlen:
            jdfn.extend(["*PAD*"] * (jlen - len(jdfn)))
    return wdfns, pdfns, jdfns, labels

def early_stop(metrics, criterion, patience):
    if not np.all(np.isnan(metrics[criterion])):
        if criterion in ['mse', 'loss_dev']:
            return np.nanargmin(metrics[criterion]) < len(metrics[criterion]) - patience
        else:
            return np.nanargmax(metrics[criterion]) < len(metrics[criterion]) - patience
    else:
        return False

def save_everything(args, exp_dir, model, metrics_dv, metrics_tr):
    metrics = {"%s" % (name):val.tolist() for (name,val) in metrics_dv.items()}
    metrics.update({"%s_tr" % (name):val.tolist() for (name,val) in metrics_tr.items()})
    with open('%s/metrics.json' % exp_dir, 'w') as of:
        json.dump(metrics, of, indent=1)
    with open('%s/params.json' % exp_dir, 'w') as of:
        json.dump({name: val for name, val in args.__dict__.items() if name != 'vis'}, of, indent=1)
    #save model
    criterion = args.criterion
    if not np.all(np.isnan(metrics_dv[criterion])):
        if criterion in ['mse', 'loss_dev'] and np.nanargmin(metrics_dv[criterion]) == len(metrics_dv[criterion]) - 1\
        or criterion not in ['mse', 'loss_dev'] and np.nanargmax(metrics_dv[criterion]) == len(metrics_dv[criterion]) - 1:
            #save state dict
            sd = model.to(torch.device('cpu')).state_dict()
            torch.save(sd, '%s/model_best_%s.pth' % (exp_dir, criterion))
            if args.gpu:
                model.to(torch.device('cuda'))

def update_metrics(metrics, golds, preds, losses, fold):
    metrics['acc'] = np.append(metrics['acc'], accuracy_score(golds, preds))
    metrics['prec'] = np.append(metrics['prec'], precision_score(golds, preds, average='weighted'))
    metrics['rec'] = np.append(metrics['rec'], recall_score(golds, preds, average='weighted'))
    metrics['f1'] = np.append(metrics['f1'], f1_score(golds, preds, average='weighted'))
    metrics['mse'] = np.append(metrics['mse'], mean_squared_error(golds, preds))
    metrics['spearman'] = np.append(metrics['spearman'], spearmanr(golds, preds)[0])
    if fold == 'DEV':
        metrics['loss_dev'] = np.append(metrics['loss_dev'], np.mean(losses))
    print("%s METRICS: %.3f & %.3f & %.3f & %.3f & %1.2f & %.3f" % \
            (fold, metrics['acc'][-1], metrics['prec'][-1], metrics['rec'][-1], metrics['f1'][-1], metrics['mse'][-1], metrics['spearman'][-1]))
    return metrics


