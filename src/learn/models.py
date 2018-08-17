import csv
import json
from math import floor
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class GELU(nn.Module):
    def forward(self, input):
        return gelu(input)


class BaseModel(nn.Module):
    def __init__(self, load_embeds, hidden_size, num_layers, nonlinearity, dropout, word2ix, binary, embed_file, embed_type, loss_fn, gpu, update_embed, only_use, comb, bbox):
        super(BaseModel, self).__init__()
        self._init(load_embeds, hidden_size, num_layers, nonlinearity, dropout, word2ix, binary, embed_file, embed_type, loss_fn, gpu, update_embed, only_use, comb, bbox)

    def _init(self, load_embeds, hidden_size, num_layers, nonlinearity, dropout, word2ix, binary, embed_file, embed_type, loss_fn, gpu, update_embed, only_use, comb, bbox):
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
        if not load_embeds:
            return

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
                self.word2vec['*PAD*'] = np.random.uniform(-0.02, 0.02, self.embed_size)
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


class DefEncoder(BaseModel):
    def __init__(self, hidden_size, bidirectional, lstm_layers, num_layers, nonlinearity, dropout, word2ix, binary, embed_file, embed_type, loss_fn, gpu, update_embed, only_use, comb, bbox):
        super(DefEncoder, self).__init__(True, hidden_size, num_layers, nonlinearity, dropout, word2ix, binary, embed_file, embed_type, loss_fn, gpu, update_embed, only_use, comb, bbox)
        self.lstm_layers = lstm_layers
        self.bidirectional = bidirectional is not None
        self.num_directions = 2 if bidirectional else 1

        #LSTMs
        self.whole_lstm = nn.LSTM(self.embed_size, floor(self.hidden_size/self.num_directions), self.lstm_layers, bidirectional=self.bidirectional)
        self.part_lstm = nn.LSTM(self.embed_size, floor(self.hidden_size/self.num_directions), self.lstm_layers, bidirectional=self.bidirectional)
        self.jj_lstm = nn.LSTM(self.embed_size, floor(self.hidden_size/self.num_directions), self.lstm_layers, bidirectional=self.bidirectional)

        #first hidden layer
        if self.num_layers > 0:
            if self.comb == 'concat':
                seq = [nn.Linear(self.hidden_size*len(self.words), self.hidden_size)]
            else:
                seq = [nn.Linear(self.hidden_size, self.hidden_size)]
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

    def forward(self, dfns, labels, bbox_fs=None, embeds=None):
        batch_sz = len(labels)
        self._refresh(batch_sz)
        #embed definitions
        wembs = []
        pembs = []
        jembs = []
        for wdef, pdef, jdef in zip(*dfns):
            wemb = [self.word2vec[w] if w in self.word2vec else self.word2vec['UNK'] for w in wdef]
            pemb = [self.word2vec[w] if w in self.word2vec else self.word2vec['UNK'] for w in pdef]
            jemb = [self.word2vec[w] if w in self.word2vec else self.word2vec['UNK'] for w in jdef]
            wembs.append(wemb)
            pembs.append(pemb)
            jembs.append(jemb)
        wembs = torch.tensor(wembs, dtype=torch.float).t()
        pembs = torch.tensor(pembs, dtype=torch.float).t()
        jembs = torch.tensor(jembs, dtype=torch.float).t()

        #apply LSTMs
        whole_out, self.whole_hidden = self.whole_lstm(wembs)
        wenc = self.whole_hidden[0][-1] if self.num_directions == 1 else self.whole_hidden[0][-2:].transpose(0,1).contiguous().view(self.batch_sz, -1)

        part_out, self.part_hidden = self.part_lstm(pembs)
        penc = self.part_hidden[0][-1] if self.num_directions == 1 else self.part_hidden[0][-2:].transpose(0,1).contiguous().view(self.batch_sz, -1)

        jj_out, self.jj_hidden = self.jj_lstm(jembs)
        jenc = self.jj_hidden[0][-1] if self.num_directions == 1 else self.jj_hidden[0][-2:].transpose(0,1).contiguous().view(self.batch_sz, -1)

        inp = torch.cat([wenc, penc, jenc], 1)

        #the rest
        logits = self.MLP(inp)
        if self.bbox and bbox_fs is not None:
            logits = torch.cat([logits, torch.Tensor(bbox_fs)], 1)
        pred = self.final(logits)
        loss = F.cross_entropy(pred, torch.LongTensor(labels).to(self.device))
        return pred, loss

    def _init_hidden(self):
        wh_0 = torch.zeros(self.num_directions*self.num_layers, self.batch_sz, floor(self.hidden_size/self.num_directions), dtype=torch.float).to(self.device)
        wc_0 = torch.zeros(self.num_directions*self.num_layers, self.batch_sz, floor(self.hidden_size/self.num_directions), dtype=torch.float).to(self.device)
        ph_0 = torch.zeros(self.num_directions*self.num_layers, self.batch_sz, floor(self.hidden_size/self.num_directions), dtype=torch.float).to(self.device)
        pc_0 = torch.zeros(self.num_directions*self.num_layers, self.batch_sz, floor(self.hidden_size/self.num_directions), dtype=torch.float).to(self.device)
        jh_0 = torch.zeros(self.num_directions*self.num_layers, self.batch_sz, floor(self.hidden_size/self.num_directions), dtype=torch.float).to(self.device)
        jc_0 = torch.zeros(self.num_directions*self.num_layers, self.batch_sz, floor(self.hidden_size/self.num_directions), dtype=torch.float).to(self.device)
        return (wh_0, wc_0), (ph_0, pc_0), (jh_0, jc_0)

    def _refresh(self, batch_sz):
        self.batch_sz = batch_sz
        self.whole_hidden, self.part_hidden, self.jj_hidden = self._init_hidden()


class TripleMLP(BaseModel):
    def __init__(self, hidden_size, num_layers, nonlinearity, dropout, word2ix, binary, embed_file, embed_type, loss_fn, gpu, update_embed, only_use, comb, bbox):
        super(TripleMLP, self).__init__(True, hidden_size, num_layers, nonlinearity, dropout, word2ix, binary, embed_file, embed_type, loss_fn, gpu, update_embed, only_use, comb, bbox)

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

    
    def forward(self, triples, labels, bbox_fs=None, embeds=None):
        #embeddings
        if not embeds:
            inp = self._get_embeddings(triples)
        else:
            import pdb; pdb.set_trace()
            inp = torch.Tensor(embeds)
        #the rest
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

    def _get_embeddings(self, triples):
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
        inp = torch.stack(inp)
        inp = F.dropout(inp, p=self.dropout)
        return inp

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


class PartWholeInteract(TripleMLP):
    def __init__(self, hidden_size, num_layers, kernel_size, nonlinearity, dropout, word2ix, binary, embed_file, embed_type, loss_fn, gpu, update_embed, only_use, comb, bbox):
        super(PartWholeInteract, self).__init__(True, hidden_size, num_layers, nonlinearity, dropout, word2ix, binary, embed_file, embed_type, loss_fn, gpu, update_embed, only_use, comb, bbox)
        self.conv = nn.Conv1d(self.embed_size, self.embed_size, kernel_size=kernel_size)
        #first hidden layer
        if self.num_layers > 0:
            if self.comb == 'concat':
                seq = [nn.Linear(self.embed_size*2, self.hidden_size)]
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

    def forward(self, triples, labels):
        #embeddings
        inp = []
        embeds = self._get_embeddings(triples)
        #part whole interaction
        #batch outer product
        embeds = [embeds[:,:self.embed_size], embeds[:,self.embed_size:self.embed_size*2], embeds[:,self.embed_size*2:]]
        inp = torch.bmm(embeds[0].unsqueeze(2), embeds[1].unsqueeze(1))
        inp = self.conv(inp)
        inp = F.max_pool1d(inp, kernel_size=inp.size(2)).squeeze()
        inp = torch.cat([inp, embeds[2]], 1)
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


