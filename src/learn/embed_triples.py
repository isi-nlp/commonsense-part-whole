"""
    Starting with elmo embeddings for a (whole, part, jj) triple, combine them with an MLP and predict
"""
import argparse, csv, itertools, json, math, os, sys, time
from collections import defaultdict

import h5py
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

EXP_DIR = '../../experiments'

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class GELU(nn.Module):
    def forward(self, input):
        return gelu(input)

class TripleDataset(Dataset):
    def __init__(self, fname, binary, only_use):
        self.triples = pd.read_csv(fname)
        self.mode = 'train'
        self.binary = binary
        self.word2ix = None
        if only_use == 'pw':
            self.words = ['whole', 'part']
        elif only_use == 'wjj':
            self.words = ['whole', 'jj']
        elif only_use == 'pjj':
            self.words = ['part', 'jj']
        else:
            self.words = ['whole', 'part', 'jj']

        #make sklearn build the vocab for me
        vectorizer = CountVectorizer(tokenizer=str.split)
        #self.triples['cat'] = self.triples.apply(lambda row: ' '.join([row['whole'], row['part'], row['jj']]), axis=1)
        self.triples['cat'] = self.triples.apply(lambda row: ' '.join([row[w] for w in self.words]), axis=1)
        feats = vectorizer.fit_transform(self.triples['cat'])
        self.word2ix = vectorizer.vocabulary_

    def set_mode(self, mode):
        self.mode = mode

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        triple = self.triples.iloc[idx][self.words].tolist()
        lname = 'bin_label' if self.binary else 'label'
        return triple, self.triples.iloc[idx][lname]

class TripleBboxDataset(TripleDataset):
    def __init__(self, fname, binary, only_use):
        super(TripleBboxDataset, self).__init__(fname, binary, only_use)
        data_path = os.path.dirname(fname)
        self.part_feats = json.load(open(os.path.join(data_path, 'part_feats.json')))
        self.whole_feats = json.load(open(os.path.join(data_path, 'whole_feats.json')))
        self.pw_feats = {pw: (feat if not np.isnan(feat) else 0.0) for pw, feat in json.load(open(os.path.join(data_path, 'pw_feats.json'))).items()}
        self._clean_nans()

    def _clean_nans(self):
        for feat in [self.part_feats, self.whole_feats]:
            for name, dct in feat.items():
                for featname, val in dct.items():
                    if np.isnan(val):
                        feat[name][featname] = 0.0

    def __getitem__(self, idx):
        triple, label = super(TripleBboxDataset, self).__getitem__(idx)
        whole, part, _ = triple
        bbox_feats = [self.whole_feats[whole]['avg_w'], self.whole_feats[whole]['avg_h'],\
                      self.part_feats[part]['avg_w'], self.part_feats[part]['avg_h'],\
                      self.pw_feats[','.join(triple[:2])]]
        return triple, label, bbox_feats

class TripleMLP(nn.Module):
    def __init__(self, hidden_size, num_layers, nonlinearity, dropout, word2ix, binary, embed_file, embed_type, loss_fn, gpu, update_embed, only_use, comb, bbox):
        super(TripleMLP, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.dropout = dropout
        self.word2ix = word2ix
        self.ix2word = {i:w for w,i in self.word2ix.items()}
        self.binary = binary
        self.embed_type = embed_type
        self.loss_fn = loss_fn
        self.device = torch.device('cuda') if gpu else torch.device('cpu')
        self.update_embed = update_embed
        self.trip_embeds = False
        if only_use == 'pw':
            self.words = ['whole', 'part']
        elif only_use == 'wjj':
            self.words = ['whole', 'jj']
        elif only_use == 'pjj':
            self.words = ['part', 'jj']
        else:
            self.words = ['whole', 'part', 'jj']
        self.comb = comb
        self.bbox = bbox

        #set up embedding layer, either with elmo or from scratch
        if embed_file:
            if embed_type == 'elmo':
                h5 = h5py.File(embed_file, 'r')
                self.word2vec = {word:vec.value[0] for word,vec in h5.items()}
                self.embed_size = 1024
            elif embed_type == 'glove' or embed_type == 'conceptnet':
                with open(embed_file, 'r') as f:
                    self.word2vec = json.load(f)
                self.embed_size = 300
                self.word2vec['UNK'] = np.random.uniform(-0.02, 0.02, self.embed_size)
            elif embed_type == 'elmo_context':
                self.trip2vec = {}
                with open(embed_file) as f:
                    r = csv.reader(f)
                    #header
                    next(r)
                    for row in r:
                        self.trip2vec[tuple(row[:3])] = torch.tensor(np.array(row[3:], dtype=np.float32)).to(self.device)
                self.trip_embeds = True
                self.embed_size = 1024

        if embed_file is None:
            self.embed_size = self.hidden_size
            self.word2vec = None

        if self.update_embed:
            self._load_pretrained()

        #first hidden layer
        if self.num_layers > 0:
            if self.comb == 'concat':
                seq = [nn.Linear(self.embed_size*len(self.words), self.hidden_size)]
            else:
                seq = [nn.Linear(self.embed_size, self.hidden_size)]
            seq = self._add_nonlinearity(seq)
            seq.append(nn.Dropout(p=self.dropout))

            #more hidden layers
            for _ in range(self.num_layers-1):
                seq.append(nn.Linear(self.hidden_size, self.hidden_size))
                seq = self._add_nonlinearity(seq)
                seq.append(nn.Dropout(p=self.dropout))
        else:
            seq = []

        #output
        if self.loss_fn == 'cross_entropy':
            out_dim = 2 if self.binary else 5
        elif self.loss_fn in ['mse', 'smooth_l1']:
            out_dim = 1
        self.MLP = nn.Sequential(*seq)
        bbox_dim = 5 if self.bbox else 0
        final_input = self.hidden_size + bbox_dim if self.num_layers > 0 else self.embed_size * len(self.words) + bbox_dim
        self.final = nn.Linear(final_input, out_dim)

    def _add_nonlinearity(self, seq):
        if self.nonlinearity == 'tanh':
            seq.append(nn.Tanh())
        elif self.nonlinearity == 'relu':
            seq.append(nn.ReLU())
        elif self.nonlinearity == 'elu':
            seq.append(nn.ELU())
        elif self.nonlinearity == 'gelu':
            seq.append(GELU())
        elif self.nonlinearity == 'selu':
            seq.append(nn.SELU())
        return seq

    def _load_pretrained(self):
        #add one for unk
        print("loading pretrained embeddings")
        self.embed = nn.Embedding(len(self.word2ix)+1, self.embed_size)
        embeddings = np.stack([self.word2vec[self.ix2word[i]] for i in range(len(self.ix2word))])
        #random init unk
        embeddings = np.concatenate([embeddings, np.random.uniform(-.2, .2, size=(1,self.embed_size))])
        self.embed.weight.data.copy_(torch.from_numpy(embeddings))

    def forward(self, triples, labels, bbox_fs=None):
        #embeddings
        inp = []
        for triple in triples:
            if self.trip_embeds:
                inp.append(self.trip2vec[tuple(triple)])
            else:
                if self.update_embed:
                    idxs = []
                    for comp in triple:
                        for c in comp.split():
                            idxs.append(self.word2ix[c] if c in self.word2ix else len(self.word2ix))
                    embeds = self.embed(torch.LongTensor(idxs).to(self.device))
                else:
                    #embeds = torch.Tensor(self.elmo_embeds[' '.join(triple)])
                    embeds = []
                    for comp in triple:
                        for c in comp.split():
                            if c in self.word2vec:
                                embeds.append(torch.Tensor(self.word2vec[c]).squeeze().to(self.device))
                            else:
                                embeds.append(torch.Tensor(self.word2vec['UNK']).squeeze().to(self.device))

                #combine multi word wholes or parts
                if ' ' in triple[0] or ' ' in triple[1]:
                    embeds = self._combine_embeds(triple, embeds)
                    if self.comb == 'concat':
                        inp.append(torch.cat(embeds))
                    elif self.comb == 'add':
                        inp.append(sum(embeds))
                    elif self.comb == 'mult':
                        inp.append(embeds[0] * embeds[1] * embeds[2])
                else:
                    if type(embeds) is list:
                        if self.comb == 'concat':
                            embeds = torch.cat(embeds)
                        elif self.comb == 'add':
                            embeds = sum(embeds)
                        elif self.comb == 'mult':
                            embeds = embeds[0] * embeds[1] * embeds[2]
                    else:
                        embeds = embeds.view(-1)
                    inp.append(embeds)

        #the rest
        inp = torch.stack(inp)
        #if self.bbox and bbox_fs is not None:
        #    inp = torch.cat([inp, torch.Tensor(bbox_fs)], 1)
        inp = F.dropout(inp, p=self.dropout)
        logits = self.MLP(inp)
        if self.bbox and bbox_fs is not None:
            logits = torch.cat([logits, torch.Tensor(bbox_fs)], 1)
        pred = self.final(logits)
        if self.loss_fn == 'cross_entropy':
            loss = F.cross_entropy(pred, torch.LongTensor(labels).to(self.device))
        elif self.loss_fn == 'mse':
            loss = F.mse_loss(pred.squeeze(), torch.Tensor(labels).to(self.device))
        elif self.loss_fn == 'smooth_l1':
            loss = F.smooth_l1_loss(pred.squeeze(), torch.Tensor(labels).to(self.device))
        return pred, loss

    def _combine_embeds(self, triple, embeds):
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
            if len(embeds) > 3:
                comb_embeds.append(embeds[3])
        else:
            comb_embeds = embeds
        return comb_embeds


#define a plotter object to make it simple to carry many window objects around
class Plotter:
    def __init__(self, args):
        import visdom
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
    return [[b[i] for b in batch] for i in range(len(batch[0]))]

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='path to train file')
    parser.add_argument('--embed-file', dest='embed_file', type=str, help='path to embeddings file. If not given, trains embeddings from scratch')
    parser.add_argument('--embed-type', dest='embed_type', choices=['elmo', 'glove', 'conceptnet', 'word2vec', 'elmo_context'], help='type of pretrained embedding to use')
    parser.add_argument('--only-use', dest='only_use', choices=['pw', 'wjj', 'pjj'], help='flag to use only two words, specifying which two words to use')
    parser.add_argument('--comb', choices=['concat', 'add', 'mult'], default='concat', help='how to combine embeddings (default: concat)')
    parser.add_argument('--epochs', type=int, default=50, help='maximum number of epochs (default: 50)')
    parser.add_argument('--hidden-size', dest='hidden_size', type=int, default=128, help='MLP hidden size (default: 128)')
    parser.add_argument('--num-layers', dest='num_layers', type=int, default=2, help='MLP number of hidden layers (default: 2; 0 = do LogReg)')
    parser.add_argument('--nonlinearity', choices=['relu', 'tanh', 'elu', 'gelu', 'selu'], default='relu', help='nonlinearity for MLP (default: relu)')
    parser.add_argument('--loss-fn', dest="loss_fn", choices=['mse', 'smooth_l1', 'cross_entropy'], default='cross_entropy', help='loss to minimize (default: cross_entropy)')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=16, help='batch size for training (default: 16)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 0.001)')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate (default: 0.2)')
    parser.add_argument('--criterion', choices=['acc', 'prec', 'rec', 'f1', 'mse', 'spearman', 'loss_dev'], default='mse', help='metric for early stopping (default: mse)')
    parser.add_argument('--patience', type=int, default=5, help='number of epochs without dev improvement in criterion metric befor early stopping (default: 5)')
    parser.add_argument('--train-check-interval', dest='train_check_interval', type=int, default=5, help='number of epochs between checking train metrics (default: 5)')
    parser.add_argument('--update-embed', dest='update_embed', action='store_const', const=True, help='flag to update ELMo embeddings (TODO)')
    parser.add_argument('--bbox-feats', dest='bbox_feats', action='store_const', const=True, help='flag to use bounding box features')
    parser.add_argument('--test-model', dest='test_model', type=str, help='path to saved model file')
    parser.add_argument('--binary', action='store_const', const=True, help='flag to predict binary labels instead of ordinal labels')
    parser.add_argument('--gpu', action='store_const', const=True, help='flag to use gpu')
    parser.add_argument('--no-plot', dest='no_plot', action='store_const', const=True, help='flag to predict NOT plot metrics')
    args = parser.parse_args()
    command = ' '.join(['python'] + sys.argv)
    args.command = command
    args.exec_time = time.strftime('%b_%d_%H:%M:%S', time.localtime())
    if not args.no_plot and not args.test_model:
        args.vis = Plotter(args)
    device = torch.device('cuda' if args.gpu else 'cpu')

    dset = TripleBboxDataset if args.bbox_feats else TripleDataset
    train_set = dset(args.file, args.binary, args.only_use)
    dev_set = dset(args.file.replace('train', 'dev'), args.binary, args.only_use)
    test_set = dset(args.file.replace('train', 'test'), args.binary, args.only_use)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=tuple_collate)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=True, collate_fn=tuple_collate)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, collate_fn=tuple_collate)

    word2ix = train_set.word2ix
    if args.embed_file is not None:
        #use all embeddings
        for dset in [dev_set, test_set]:
            for word, ix in dset.word2ix.items():
                if word not in word2ix:
                    word2ix[word] = len(word2ix)

    torch.manual_seed(4746)
    model = TripleMLP(args.hidden_size, args.num_layers, args.nonlinearity, args.dropout, word2ix, args.binary, args.embed_file, args.embed_type, args.loss_fn, args.gpu, args.update_embed, args.only_use, args.comb, args.bbox_feats)
    print(model)
    if args.test_model:
        sd = torch.load(args.test_model)
        model.load_state_dict(sd)
        dont_train = True
    else:
        dont_train = False
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    metrics_tr = defaultdict(lambda: np.array([]))
    metrics_dv = defaultdict(lambda: np.array([]))
    for epoch in range(args.epochs):

        losses, losses_dv = [], []
        #TRAIN LOOP
        if not dont_train:
            model.train()
            train_golds, train_preds = [], []
            try:
                for batch_ix, data in tqdm(enumerate(train_loader)):
                    optimizer.zero_grad()
                    if args.bbox_feats:
                        triples, labels, bbox_fs = data
                        preds, loss = model(triples, labels, bbox_fs=bbox_fs)
                    else:
                        triples, labels = data
                        preds, loss = model(triples, labels)
                    train_golds.extend(labels)
                    if args.loss_fn == 'cross_entropy':
                        train_preds.extend([pred.argmax().item() for pred in preds])
                    elif args.loss_fn in ['mse', 'smooth_l1']:
                        train_preds.extend([pred.round().long().item() for pred in preds])

                    loss.backward()
                    if torch.isnan(model.MLP[0].weight.grad).any().item() == 1:
                        import pdb; pdb.set_trace()
                    optimizer.step()
                    losses.append(loss.item())
                    if not args.no_plot:
                        args.vis.plot_batch_loss(losses, 10)
            except KeyboardInterrupt:
                import pdb; pdb.set_trace()
            metrics_tr['loss'] = np.append(metrics_tr['loss'], np.mean(losses))
            print("Loss: %.4f" % np.mean(losses))

        #DEV LOOP
        with torch.no_grad():
            model.eval()
            dev_trips, dev_golds, dev_preds = [], [], []
            for batch_ix, data in tqdm(enumerate(dev_loader)):
                if args.bbox_feats:
                    triples, labels, bbox_fs = data
                    preds, loss = model(triples, labels, bbox_fs=bbox_fs)
                else:
                    triples, labels = data
                    preds, loss = model(triples, labels)
                dev_trips.extend(triples)
                dev_golds.extend(labels)
                if args.loss_fn == 'cross_entropy':
                    dev_preds.extend([pred.argmax().item() for pred in preds])
                elif args.loss_fn in ['mse', 'smooth_l1']:
                    dev_preds.extend([pred.round().long().item() for pred in preds])
                losses_dv.append(loss)

        #SAVE AND PRINT STUFF
        if epoch % args.train_check_interval == 0 and not dont_train:
            metrics_tr = update_metrics(metrics_tr, train_golds, train_preds, losses, 'TRAIN')
        metrics_dv = update_metrics(metrics_dv, dev_golds, dev_preds, losses_dv, 'DEV')

        if epoch == 0 and not dont_train:
            #make experiment directory
            exp_dir = os.path.join(EXP_DIR, '_'.join(['embed', args.exec_time]))
            print("output directory: %s" % exp_dir)
            os.mkdir(exp_dir)
        elif args.test_model:
            exp_dir = os.path.dirname(args.test_model)
        save_everything(args, exp_dir, model, metrics_dv, metrics_tr)

        #PLOT STUFF
        if not args.no_plot:
            if epoch == 0 and not dont_train:
                args.vis.populate(metrics_dv, metrics_tr)
            elif dont_train:
                #write preds
                with open('%s/dev_preds.csv' % exp_dir, 'w') as of:
                    w = csv.writer(of)
                    lname = 'bin_label' if args.binary else 'label'
                    w.writerow(train_set.words + ['pred', lname])
                    for trip, pred, gold in zip(dev_trips, dev_preds, dev_golds):
                        w.writerow([*trip, pred, gold])

                #visualize confusion matrix
                conmat = confusion_matrix(dev_golds, dev_preds)
                labels = ['impossible', 'unlikely', 'unrelated', 'probably', 'guaranteed'] if not args.binary else ['non-entailment', 'entailment']
                ticks = np.arange(len(labels))
                plt.figure()

                cm = conmat.copy()
                for i in range(len(labels)):
                    cm[i][i] = 0
                plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                plt.xticks(ticks, labels, rotation=45)
                plt.yticks(ticks, labels)
                thresh = cm.max() / 2
                for i,j in itertools.product(range(len(labels)), range(len(labels))):
                    plt.text(j,i,conmat[i][j],horizontalalignment='center',color='white' if cm[i][j] > thresh else 'black')

                #df_cm = pd.DataFrame(conmat, index=labels, columns=labels)
                #sns.heatmap(df_cm, annot=True, fmt='g', norm=colors.LogNorm(vmin=0,vmax=conmat.max()), cbar=False, cmap='hot', annot_kws={'fontsize': 'xx-large'})
                #sns.heatmap(df_cm, annot=True, fmt='g', cbar=False, annot_kws={'fontsize': 'xx-large'})
                embed_type = args.embed_type if args.embed_type is not None else 'scratch'
                plt.title('Confusion matrix: {} embeddings'.format(embed_type))
                plt.xlabel('Predicted class')
                plt.ylabel('True class')
                plt.tight_layout()
                plt.show()
                print(f"model saved at {exp_dir}")
                sys.exit(0)
            else:
                args.vis.update(epoch, metrics_dv, metrics_tr)

        if early_stop(metrics_dv, args.criterion, args.patience):
            print("early stopping point hit")
            #reload best model
            model = TripleMLP(args.hidden_size, args.num_layers, args.nonlinearity, args.dropout, word2ix, args.binary, args.embed_file, args.embed_type, args.loss_fn, args.gpu, args.update_embed, args.only_use, args.comb, args.bbox_feats)
            sd = torch.load('%s/model_best_%s.pth' % (exp_dir, args.criterion))
            model.load_state_dict(sd)
            model.to(device)
            #flag to rerun on dev
            dont_train = True

