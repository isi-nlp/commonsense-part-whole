"""
    Starting with elmo embeddings for a (whole, part, jj) triple, combine them with an MLP and predict
"""
import argparse
import csv
import itertools
import json
import math
import os
import sys
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import datasets
import models
import plotter
import utils

EXP_DIR = '../../experiments'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='path to train file')
    parser.add_argument('--model', choices=['MLP', 'pwi', 'definitions'], default="MLP", help="which model to train (default: MLP)")
    parser.add_argument('--embed-file', dest='embed_file', type=str, help='path to embeddings file. If not given, trains embeddings from scratch')
    parser.add_argument('--embed-type', dest='embed_type', choices=['elmo', 'glove', 'conceptnet', 'word2vec', 'elmo_context'], help='type of pretrained embedding to use')
    parser.add_argument('--only-use', dest='only_use', choices=['pw', 'wjj', 'pjj'], help='flag to use only two words, specifying which two words to use')
    parser.add_argument('--comb', choices=['concat', 'add', 'mult'], default='concat', help='how to combine embeddings (default: concat)')
    parser.add_argument('--epochs', type=int, default=50, help='maximum number of epochs (default: 50)')
    parser.add_argument('--lstm-layers', dest='lstm_layers', type=int, default=1, help='number of layers in the lstm encoders')
    parser.add_argument('--bidirectional', action='store_const', const=True, help='flag to use bilstm encoders')
    parser.add_argument('--kernel-size', dest='kernel_size', type=int, default=3, help='kernel size for conv over part-whole interaction (default: 3)')
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
        args.vis = plotter.Plotter(args)
    device = torch.device('cuda' if args.gpu else 'cpu')

    ### DATA LOADERS
    trip2embeds = defaultdict(list)
    if args.bbox_feats:
        dset = datasets.TripleBboxDataset
        collate_fn = utils.tuple_collate
    elif args.model == 'definitions':
        dset = datasets.DefinitionDataset
        collate_fn = utils.dfn_collate
    elif args.embed_type == 'elmo_context' and 'retr' in args.embed_file:
        print("loading elmo retrieved embeds")
        with open(args.embed_file) as f:
            r = csv.reader(f)
            next(r)
            for row in tqdm(r):
                trip2embeds[tuple(row[:3])] = np.array(row[4:])
        dset = datasets.TripleRetrDataset
        collate_fn = utils.tuple_collate
    else:
        dset = datasets.TripleDataset
        collate_fn = utils.tuple_collate
    train_set = dset(args.file, args.binary, args.only_use, trip2embeds=trip2embeds)
    dev_set = dset(args.file.replace('train', 'dev'), args.binary, args.only_use, trip2embeds=trip2embeds)
    test_set = dset(args.file.replace('train', 'test'), args.binary, args.only_use, trip2embeds=trip2embeds)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    #memory
    del trip2embeds

    #combine word lookups
    word2ix = train_set.word2ix
    if args.embed_file is not None:
        #use all embeddings
        for dset in [dev_set, test_set]:
            for word, ix in dset.word2ix.items():
                if word not in word2ix:
                    word2ix[word] = len(word2ix)

    torch.manual_seed(4746)

    ### MODEL SETUP
    if args.model == 'MLP':
        model = models.TripleMLP(args.hidden_size, args.num_layers, args.nonlinearity, args.dropout, word2ix, args.binary, args.embed_file, args.embed_type, args.loss_fn, args.gpu, args.update_embed, args.only_use, args.comb, args.bbox_feats)
    elif args.model == 'pwi':
        model = models.PartWholeInteract(args.hidden_size, args.num_layers, args.kernel_size, args.nonlinearity, args.dropout, word2ix, args.binary, args.embed_file, args.embed_type, args.loss_fn, args.gpu, args.update_embed, args.only_use, args.comb, args.bbox_feats)
    elif args.model == 'definitions':
        model = models.DefEncoder(args.hidden_size, args.bidirectional, args.lstm_layers, args.num_layers, args.nonlinearity, args.dropout, word2ix, args.binary, args.embed_file, args.embed_type, args.loss_fn, args.gpu, args.update_embed, args.only_use, args.comb, args.bbox_feats)
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

    ### MAIN LOOP
    for epoch in range(args.epochs):

        losses, losses_dv = [], []
        #TRAIN LOOP
        if not dont_train:
            model.train()
            train_golds, train_preds = [], []
            try:
                for batch_ix, data in tqdm(enumerate(train_loader)):
                    optimizer.zero_grad()
                    if isinstance(train_set, datasets.TripleBboxDataset):
                        triples, labels, bbox_fs = data
                        preds, loss = model(triples, labels, bbox_fs=bbox_fs)
                    elif isinstance(train_set, datasets.TripleRetrDataset):
                        triples, labels, embeds = data
                        preds, loss = model(triples, labels, embeds=embeds)
                    elif isinstance(train_set, datasets.DefinitionDataset):
                        *dfns, labels = data
                        preds, loss = model(dfns, labels)
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
                if isinstance(train_set, datasets.TripleBboxDataset):
                    triples, labels, bbox_fs = data
                    preds, loss = model(triples, labels, bbox_fs=bbox_fs)
                elif isinstance(train_set, datasets.DefinitionDataset):
                    *dfns, labels = data
                    preds, loss = model(dfns, labels)
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
            metrics_tr = utils.update_metrics(metrics_tr, train_golds, train_preds, losses, 'TRAIN')
        metrics_dv = utils.update_metrics(metrics_dv, dev_golds, dev_preds, losses_dv, 'DEV')

        if epoch == 0 and not dont_train:
            #make experiment directory
            exp_dir = os.path.join(EXP_DIR, '_'.join(['embed', args.exec_time]))
            print("output directory: %s" % exp_dir)
            os.mkdir(exp_dir)
        elif args.test_model:
            exp_dir = os.path.dirname(args.test_model)
        utils.save_everything(args, exp_dir, model, metrics_dv, metrics_tr)

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

        # EARLY STOPPING
        if utils.early_stop(metrics_dv, args.criterion, args.patience):
            print("early stopping point hit")
            #reload best model
            if args.model == 'MLP':
                model = models.TripleMLP(args.hidden_size, args.num_layers, args.nonlinearity, args.dropout, word2ix, args.binary, args.embed_file, args.embed_type, args.loss_fn, args.gpu, args.update_embed, args.only_use, args.comb, args.bbox_feats)
            elif args.model == 'pwi':
                model = models.PartWholeInteract(args.hidden_size, args.num_layers, args.kernel_size, args.nonlinearity, args.dropout, word2ix, args.binary, args.embed_file, args.embed_type, args.loss_fn, args.gpu, args.update_embed, args.only_use, args.comb, args.bbox_feats)
            sd = torch.load('%s/model_best_%s.pth' % (exp_dir, args.criterion))
            model.load_state_dict(sd)
            model.to(device)
            #flag to rerun on dev
            dont_train = True
