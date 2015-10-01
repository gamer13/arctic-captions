import cPickle as pkl
import gzip
import os
import sys
import time
import h5py

import numpy

from IPython import embed


def prepare_data(caps, features, worddict, maxlen=None, n_words=10000, zero_pad=False):
    # x: a list of sentences
    seqs = [] # List of captions
    feat_list = [] # List of visual features
    for cc in caps:
        seqs.append([worddict[w.lower()] if worddict[w.lower()] < n_words else 1 for w in cc[0].split()])
        id_for_features = features['lookup'][cc[1]]
        feat_list.append(features['feats'][id_for_features]) # Get features for the captions

    lengths = [len(s) for s in seqs] # Get lengths of captions

    if maxlen != None:
        new_seqs = []
        new_feat_list = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, feat_list):
            # Filter captions on max length
            if l < maxlen:
                new_seqs.append(s)
                new_feat_list.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        feat_list = new_feat_list
        seqs = new_seqs

        # ... in case there is no data left
        if len(lengths) < 1:
            return None, None, None

    # Construct visual features array
    # Numpy array of size (n_samples, 100352 (512 * 14 * 14))
    y = numpy.zeros((len(feat_list), feat_list[0].shape[0])).astype('float32')
    for idx, ff in enumerate(feat_list):
        y[idx,:] = ff # Add to numpy array
    y = y.reshape([y.shape[0], 14*14, 512])  # Reshape to (n_samples, 14*14, 512)
    if zero_pad:
        # Zero padding
        y_pad = numpy.zeros((y.shape[0], y.shape[1]+1, y.shape[2])).astype('float32')
        y_pad[:,:-1,:] = y
        y = y_pad

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)+1

    x = numpy.zeros((maxlen, n_samples)).astype('int64')        # Construct captions array
    x_mask = numpy.zeros((maxlen, n_samples)).astype('float32') # Construct captions mask array
    for idx, s in enumerate(seqs):
        x[:lengths[idx],idx] = s
        x_mask[:lengths[idx]+1,idx] = 1.

    return x, x_mask, y

def load_data(load_train=True, load_dev=True, load_test=True, path='/media/Data/flipvanrijn/datasets/coco/'):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset
    '''

    #############
    # LOAD DATA #
    #############

    print '... loading data'

    if load_train:
        f_train = h5py.File(path+'coco_train.h5', 'r')
        train_cap = zip(f_train['captions/caps'], f_train['captions/img_ids'])
        train = (train_cap, f_train['images'])
    else:
        train = None
    if load_test:
        f_test = h5py.File(path+'coco_test.h5', 'r')
        test_cap = zip(f_test['captions/caps'], f_test['captions/img_ids'])
        test = (test_cap, f_test['images'])
    else:
        test = None
    if load_dev:
        f_dev = h5py.File(path+'coco_dev.h5', 'r')
        dev_cap = zip(f_dev['captions/caps'], f_dev['captions/img_ids'])
        valid = (dev_cap, f_dev['images'])

    with open(path+'dictionary.pkl', 'rb') as f:
        worddict = pkl.load(f)

    return train, valid, test, worddict
