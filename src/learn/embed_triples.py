"""
    Starting with elmo embeddings for a (whole, part, jj) triple, combine them with an MLP and predict
    TODO: reload best model after early stopping and predict on test
"""
import argparse, csv, json, os, sys, time
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import visdom

EXP_DIR = '/home/jamesm/commonsense-part-whole/experiments'

class TripleDataset(Dataset):
    def __init__(self, fname, binary=False, vocab=False):
        self.triples = pd.read_csv(fname)
        self.mode = 'train'
        self.binary = binary
        self.word2ix = None
        if vocab:
            #make sklearn build the vocab for me
            vectorizer = CountVectorizer(tokenizer=str.split)
            self.triples['cat'] = self.triples.apply(lambda row: ' '.join([row['whole'], row['part'], row['jj']]), axis=1)
            feats = vectorizer.fit_transform(self.triples['cat'])
            self.word2ix = vectorizer.vocabulary_

    def set_mode(self, mode):
        self.mode = mode

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        triple = self.triples.iloc[idx][['whole', 'part', 'jj']].tolist()
        lname = 'bin_label' if self.binary else 'label'
        return triple, self.triples.iloc[idx][lname]

class TripleMLP(nn.Module):
    def __init__(self, hidden_size, num_layers, nonlinearity, dropout, binary=False, embed_file=None, embed_type=None, word2ix=None, loss_fn='cross_entropy'):
        super(TripleMLP, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.dropout = dropout
        self.binary = binary
        self.embed_type = embed_type
        self.loss_fn = loss_fn

        #set up embedding layer, either with elmo or from scratch
        if embed_file:
            if embed_type == 'elmo':
                self.elmo_embeds = h5py.File(embed_file, 'r')
                self.embed_size = 1024
            elif embed_type == 'glove':
                with open(embed_file, 'r') as f:
                    self.glove_embeds = json.load(f)
                self.embed_size = 300
        else:
            #add one for unk
            self.embed = nn.Embedding(len(word2ix)+1, self.hidden_size)
            self.embed_size = self.hidden_size
            self.word2ix = word2ix

        #first hidden layer
        if self.num_layers > 0:
            seq = [nn.Linear(self.embed_size*3, self.hidden_size)]
            if self.nonlinearity == 'tanh':
                seq.append(nn.Tanh())
            else:
                seq.append(nn.ReLU())
            seq.append(nn.Dropout(p=self.dropout))

            #more hidden layers
            for _ in range(self.num_layers-1):
                seq.append(nn.Linear(self.hidden_size, self.hidden_size))
                if self.nonlinearity == 'tanh':
                    seq.append(nn.Tanh())
                else:
                    seq.append(nn.ReLU())
                seq.append(nn.Dropout(p=self.dropout))
        else:
            seq = []

        #output
        if self.loss_fn == 'cross_entropy':
            out_dim = 2 if self.binary else 5
        elif self.loss_fn in ['mse', 'smooth_l1']:
            out_dim = 1
        final_input = self.hidden_size if self.num_layers > 0 else self.embed_size * 3
        seq.append(nn.Linear(final_input, out_dim))
        #seq.append(nn.LogSoftmax())
        self.MLP = nn.Sequential(*seq)

    def forward(self, triples, labels):
        #embeddings...
        inp = []
        for triple in triples:
            if self.embed_type == 'elmo':
                #embeds = torch.Tensor(self.elmo_embeds[' '.join(triple)])
                embeds = []
                for comp in triple:
                    for c in comp.split():
                        embeds.append(torch.Tensor(self.elmo_embeds[c].value).squeeze())
            elif self.embed_type == 'glove':
                embeds = []
                for comp in triple:
                    for c in comp.split():
                        embeds.append(torch.Tensor(self.glove_embeds[c]))
            else:
                idxs = []
                for comp in triple:
                    for c in comp.split():
                        idxs.append(self.word2ix[c] if c in self.word2ix else len(self.word2ix))
                embeds = self.embed(torch.LongTensor(idxs))

            #combine multi word wholes or parts
            if ' ' in triple[0] or ' ' in triple[1]:
                embeds = self.combine_embeds(triple, embeds)
                inp.append(torch.cat(embeds))
            else:
                if type(embeds) is list:
                    embeds = torch.cat(embeds)
                else:
                    embeds = embeds.view(-1)
                inp.append(embeds)

        #the rest
        inp = F.dropout(torch.stack(inp), p=self.dropout)
        pred = self.MLP(inp)
        if self.loss_fn == 'cross_entropy':
            loss = F.cross_entropy(pred, torch.LongTensor(labels))
        elif self.loss_fn == 'mse':
            loss = F.mse_loss(pred.squeeze(), torch.Tensor(labels))
        elif self.loss_fn == 'smooth_l1':
            loss = F.smooth_l1_loss(pred.squeeze(), torch.Tensor(labels))
        return pred, loss

    def combine_embeds(self, triple, embeds):
        #just some hard coding...
        comb_embeds = []
        if ' ' in triple[0]:
            comb_embeds.append((embeds[0]+embeds[1])/2)
            if ' 'in triple[1]:
                #this doesn't happen :)
                pass
            comb_embeds.extend(embeds[2:])
        elif ' ' in triple[1]:
            comb_embeds.append(embeds[0])
            comb_embeds.append((embeds[1]+embeds[2])/2)
            comb_embeds.append(embeds[3])
        else:
            comb_embeds = embeds
        return comb_embeds

#define a plotter object to make it simple to carry many window objects around
class Plotter:
    def __init__(self, args):
        self.vis = visdom.Visdom(env="%s" % args.exec_time)
        self.batch_loss_plt = self.vis.line(np.array([[0, 0]]), np.array([[0, 0]]),
                                            opts={'title': "Batch loss",
                                                "xlabel": "Batch",
                                                "legend": ["loss", "moving average"],
                                                "showlegend": True})
        self.show_hps(args.__dict__)

    def populate(self, metrics_dv, metrics_tr):
        x = np.array([0])

        self.dv_prec_plt = self.vis.line(metrics_dv['prec'][-1:], x,
                            opts={'title': "Dev precision", "xlabel": "Epoch"})
        self.dv_rec_plt = self.vis.line(metrics_dv['rec'][-1:], x,
                            opts={'title': "Dev recall", "xlabel": "Epoch"})
        self.dv_f1_plt = self.vis.line(metrics_dv['f1'][-1:], x,
                            opts={'title': "Dev F1", "xlabel": "Epoch"})
        self.dv_mse_plt = self.vis.line(metrics_dv['mse'][-1:], x,
                            opts={'title': "Dev MSE", "xlabel": "Epoch"})
        self.dv_rho_plt = self.vis.line(metrics_dv['spearman'][-1:], x,
                            opts={'title': "Dev spearman's rho", "xlabel": "Epoch"})
        self.tr_loss_plt = self.vis.line(metrics_tr['loss'][-1:], x,
                            opts={'title': "Train loss", "xlabel": "Epoch"})
        self.dv_loss_plt = self.vis.line(metrics_dv['loss_dev'][-1:], x,
                            opts={'title': "Dev loss", "xlabel": "Epoch"})

    def update(self, epoch, metrics_dv, metrics_tr):
        x = np.array([epoch])

        self.vis.line(metrics_dv['prec'][-1:], x, win=self.dv_prec_plt, update='append')
        self.vis.line(metrics_dv['rec'][-1:], x, win=self.dv_rec_plt, update='append')
        self.vis.line(metrics_dv['f1'][-1:], x, win=self.dv_f1_plt, update='append')
        self.vis.line(metrics_dv['mse'][-1:], x, win=self.dv_mse_plt, update='append')
        self.vis.line(metrics_dv['spearman'][-1:], x, win=self.dv_rho_plt, update='append')
        self.vis.line(metrics_dv['loss_dev'][-1:], x, win=self.dv_loss_plt, update='append')
        self.vis.line(metrics_tr['loss'][-1:], x, win=self.tr_loss_plt, update='append')

    def plot_batch_loss(self, losses, avg_run):
        x = np.arange(len(losses))
        l_avg = [np.mean(losses[max(i-avg_run,0):i+1]) for i in range(len(losses))]
        data = np.vstack([losses, l_avg]).transpose()
        self.vis.line(data, x, win=self.batch_loss_plt, update='new')

    def show_hps(self, params):
        self.text = self.vis.text("All hyperparameters:")
        for key, val in params.items():
            self.vis.text("%s: %s" % (str(key), str(val)), win=self.text, append=True)

def tuple_collate(batch):
    return [b[0] for b in batch], [b[1] for b in batch]

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
        if np.nanargmax(metrics_dv[criterion]) == len(metrics_dv[criterion]) - 1:
            #save state dict
            sd = model.to(torch.device('cpu')).state_dict()
            torch.save(sd, '%s/model_best_%s.pth' % (exp_dir, criterion))

def update_metrics(metrics, golds, preds, losses, fold):
    metrics['acc'] = np.append(metrics['acc'], accuracy_score(golds, preds))
    metrics['prec'] = np.append(metrics['prec'], precision_score(golds, preds, average='weighted'))
    metrics['rec'] = np.append(metrics['rec'], recall_score(golds, preds, average='weighted'))
    metrics['f1'] = np.append(metrics['f1'], f1_score(golds, preds, average='weighted'))
    metrics['mse'] = np.append(metrics['mse'], mean_squared_error(golds, preds))
    metrics['spearman'] = np.append(metrics['spearman'], spearmanr(golds, preds)[0])
    if fold == 'DEV':
        metrics['loss_dev'] = np.append(metrics['loss_dev'], np.mean(losses))
    print("%s METRICS: %.3f & %.3f & %.3f & %.3f & %.3f & %.3f" % \
            (fold, metrics['acc'][-1], metrics['prec'][-1], metrics['rec'][-1], metrics['f1'][-1], metrics['mse'][-1], metrics['spearman'][-1]))
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='path to train file')
    parser.add_argument('--embed-file', dest='embed_file', type=str, help='path to embeddings file. If not given, trains embeddings from scratch')
    parser.add_argument('--embed-type', dest='embed_type', choices=['elmo', 'glove', 'word2vec'], help='type of pretrained embedding to use')
    parser.add_argument('--epochs', type=int, default=50, help='maximum number of epochs (default: 50)')
    parser.add_argument('--hidden-size', dest='hidden_size', type=int, default=128, help='MLP hidden size (default: 128)')
    parser.add_argument('--num-layers', dest='num_layers', type=int, default=2, help='MLP number of hidden layers (default: 2; 0 = do LogReg)')
    parser.add_argument('--nonlinearity', choices=['relu', 'tanh'], default='relu', help='nonlinearity for MLP (default: tanh)')
    parser.add_argument('--loss-fn', dest="loss_fn", choices=['mse', 'smooth_l1', 'cross_entropy'], default='cross_entropy', help='loss to minimize (default: cross_entropy)')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=16, help='batch size for training (default: 16)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 0.001)')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate (default: 0.2)')
    parser.add_argument('--criterion', choices=['acc', 'prec', 'rec', 'f1', 'mse', 'spearman', 'loss_dev'], default='mse', help='metric for early stopping (default: mse)')
    parser.add_argument('--patience', type=int, default=5, help='number of epochs without dev improvement in criterion metric befor early stopping (default: 5)')
    parser.add_argument('--train-check-interval', dest='train_check_interval', type=int, default=5, help='number of epochs between checking train metrics (default: 5)')
    parser.add_argument('--update-embed', dest='update_embed', action='store_const', const=True, help='flag to update ELMo embeddings (TODO)')
    parser.add_argument('--binary', action='store_const', const=True, help='flag to predict binary labels instead of ordinal labels')
    parser.add_argument('--no-plot', dest='no_plot', action='store_const', const=True, help='flag to predict NOT plot metrics')
    args = parser.parse_args()
    command = ' '.join(['python'] + sys.argv)
    args.command = command
    args.exec_time = time.strftime('%b_%d_%H:%M:%S', time.localtime())
    if not args.no_plot:
        args.vis = Plotter(args)

    build_vocab = args.embed_file is None
    train_set = TripleDataset(args.file, args.binary, vocab=build_vocab)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=tuple_collate)
    dev_set = TripleDataset(args.file.replace('train', 'dev'), args.binary)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=True, collate_fn=tuple_collate)
    test_set = TripleDataset(args.file.replace('train', 'test'), args.binary)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, collate_fn=tuple_collate)

    word2ix = None
    if not args.embed_file:
        word2ix = train_set.word2ix

    torch.manual_seed(4746)
    model = TripleMLP(args.hidden_size, args.num_layers, args.nonlinearity, args.dropout, binary=args.binary, embed_file=args.embed_file, embed_type=args.embed_type, word2ix=word2ix, loss_fn=args.loss_fn)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    metrics_tr = defaultdict(lambda: np.array([]))
    metrics_dv = defaultdict(lambda: np.array([]))
    for epoch in range(args.epochs):

        #TRAIN LOOP
        losses, losses_dv = [], []
        model.train()
        train_golds, train_preds = [], []
        for batch_ix, (triples, labels) in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()
            preds, loss = model(triples, labels)
            train_golds.extend(labels)
            if args.loss_fn == 'cross_entropy':
                train_preds.extend([pred.argmax().item() for pred in preds])
            elif args.loss_fn in ['mse', 'smooth_l1']:
                train_preds.extend([pred.round().long().item() for pred in preds])

            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if not args.no_plot:
                args.vis.plot_batch_loss(losses, 10)
        metrics_tr['loss'] = np.append(metrics_tr['loss'], np.mean(losses))

        #DEV LOOP
        print("Loss: %.4f" % np.mean(losses))
        with torch.no_grad():
            model.eval()
            dev_golds, dev_preds = [], []
            for batch_ix, (triples, labels) in tqdm(enumerate(dev_loader)):
                preds, loss = model(triples, labels)
                dev_golds.extend(labels)
                if args.loss_fn == 'cross_entropy':
                    dev_preds.extend([pred.argmax().item() for pred in preds])
                elif args.loss_fn in ['mse', 'smooth_l1']:
                    dev_preds.extend([pred.round().long().item() for pred in preds])
                losses_dv.append(loss)

        #SAVE AND PRINT STUFF
        if epoch % args.train_check_interval == 0:
            metrics_tr = update_metrics(metrics_tr, train_golds, train_preds, losses, 'TRAIN')
        metrics_dv = update_metrics(metrics_dv, dev_golds, dev_preds, losses_dv, 'DEV')

        if epoch == 0:
            #make experiment directory
            exp_dir = os.path.join(EXP_DIR, '_'.join(['embed', args.exec_time]))
            print("output directory: %s" % exp_dir)
            os.mkdir(exp_dir)
        save_everything(args, exp_dir, model, metrics_dv, metrics_tr)

        #PLOT STUFF
        if not args.no_plot:
            if epoch == 0:
                args.vis.populate(metrics_dv, metrics_tr)
            else:
                args.vis.update(epoch, metrics_dv, metrics_tr)

        if early_stop(metrics_dv, args.criterion, args.patience):
            print("early stopping point hit")
            sys.exit(0)

