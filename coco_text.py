import cPickle as pkl
import gzip
import os
import sys
import time

import numpy
import scipy.sparse

from IPython import embed

def prepare_data(caps, features, contexts, worddict, maxlen=None, n_words=10000, zero_pad=False, tex_dim=None):
    # x: a list of sentences
    seqs = []
    feat_list = []
    context_list = []
    for cc in caps:
        seqs.append([worddict[w] if worddict[w] < n_words else 1 for w in cc[0].split()])
        feat_list.append(features[cc[1]])
        context_list.append(contexts[cc[1]])

    lengths = [len(s) for s in seqs]

    if maxlen != None:
        new_seqs = []
        new_feat_list = []
        new_context_list = []
        new_lengths = []
        for l, s, y, c in zip(lengths, seqs, feat_list, context_list):
            if l < maxlen:
                new_seqs.append(s)
                new_feat_list.append(y)
                new_context_list.append(c)
                new_lengths.append(l)
        lengths = new_lengths
        feat_list = new_feat_list
        context_list = new_context_list
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None, None, None

    # img feats
    y = numpy.zeros((len(feat_list), feat_list[0].shape[1])).astype('float32')
    for idx, ff in enumerate(feat_list):
        y[idx,:] = numpy.array(ff.todense())
    y = y.reshape([y.shape[0], 14*14, 512])
    if zero_pad:
        y_pad = numpy.zeros((y.shape[0], y.shape[1]+1, y.shape[2])).astype('float32')
        y_pad[:,:-1,:] = y
        y = y_pad

    # context feats
    c = numpy.zeros((len(context_list), context_list[0].shape[1])).astype('float32')
    for idx, cc in enumerate(context_list):
        if scipy.sparse.issparse(cc):
            c[idx,:] = numpy.array(cc.todense())
        else:
            c[idx,:] = numpy.array(cc)

    # infer shape for the context (different for some experiments)
    # TODO: Make this more robust, now assumes either a vector or a matrix shape

    if tex_dim != 512:
        c = c.reshape([c.shape[0], 1, tex_dim])
    else:
        c = c.reshape([c.shape[0], c.shape[1] / tex_dim, tex_dim])
    if zero_pad:
        c_pad = numpy.zeros((c.shape[0], c.shape[1]+1, c.shape[2])).astype('float32')
        c_pad[:,:-1,:] = c
        c = c_pad   

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)+1

    x = numpy.zeros((maxlen, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen, n_samples)).astype('float32')
    for idx, s in enumerate(seqs):
        x[:lengths[idx],idx] = s
        x_mask[:lengths[idx]+1,idx] = 1.

    return x, x_mask, y, c

def load_data(load_train=True, load_dev=True, load_test=True, path='./'):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset
    '''

    #############
    # LOAD DATA #
    #############

    print '... loading data'

    train = None
    test  = None
    valid = None

    if load_train:
        with open(path+'/coco_align.train.pkl', 'rb') as f:
            train_cap  = pkl.load(f)
            train_feat = pkl.load(f)
            train_ctx  = pkl.load(f)
        train = (train_cap, train_feat, train_ctx)
    if load_test:
        with open(path+'/coco_align.test.pkl', 'rb') as f:
            test_cap  = pkl.load(f)
            test_feat = pkl.load(f)
            test_ctx  = pkl.load(f)
        test = (test_cap, test_feat, test_ctx)
    if load_dev:
        with open(path+'/coco_align.dev.pkl', 'rb') as f:
            dev_cap  = pkl.load(f)
            dev_feat = pkl.load(f)
            dev_ctx  = pkl.load(f)
        valid = (dev_cap, dev_feat, dev_ctx)

    with open(path+'/dictionary.pkl', 'rb') as f:
        worddict = pkl.load(f)

    return train, valid, test, worddict

