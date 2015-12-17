"""
Sampling script for attention models

Works on CPU with support for multi-process
"""
import argparse
import numpy
import cPickle as pkl
import json
from progress.bar import Bar
import scipy.sparse

from capgen_text import build_sampler, gen_sample, \
                   load_params, \
                   init_params, \
                   init_tparams, \
                   get_dataset \

import theano
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from IPython import embed

def create_sample(tparams, f_init, f_next, context, text, options, trng, k, normalize):
    sample, score = gen_sample(tparams, f_init, f_next, context, text, options,
                               trng=trng, k=k, maxlen=200, stochastic=False)
    # adjust for length bias
    if normalize:
        lengths = numpy.array([len(s) for s in sample])
        score = score / lengths
    sidx = numpy.argmin(score)
    return sample[sidx]

def main(model, saveto, path='./', k=1, normalize=False, zero_pad=False, datasets='dev,test', sampling=False, pkl_name=None):
    # load model model_options
    if pkl_name is None:
        pkl_name = model
    with open('%s.pkl'% pkl_name, 'rb') as f:
        options = pkl.load(f)

    # fetch data, skip ones we aren't using to save time
    load_data, prepare_data = get_dataset(options['dataset'])
    _, valid, test, worddict = load_data(load_train=False, load_dev=True if 'dev' in datasets else False,
                                             load_test=True if 'test' in datasets else False, path=path)

    # <eos> means end of sequence (aka periods), UNK means unknown
    word_idict = dict()
    for kk, vv in worddict.iteritems():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'

    # build sampler
    trng = RandomStreams(1234)
    # this is zero indicate we are not using dropout in the graph
    use_noise = theano.shared(numpy.float32(0.), name='use_noise')

    # get the parameters
    params = init_params(options)
    params = load_params(model, params)
    tparams = init_tparams(params)

    # build the sampling computational graph
    # see capgen.py for more detailed explanations
    f_init, f_next = build_sampler(tparams, options, use_noise, trng, sampling=sampling)

    # index -> words
    def _seqs2words(cc):
        ww = []
        for w in cc:
            if w == 0:
                break
            ww.append(word_idict[w])
        if ww[-1] != '.':
            ww.append('.')
        return ' '.join(ww)

    # unsparsify, reshape, and queue
    def _send_job(context, text):
        textdim = text.todense().shape[1] / 512
        cc = context.todense().reshape([14*14,512])
        if scipy.sparse.issparse(text):
            tt = text.todense().reshape([textdim, 512])
        else:
            tt = text.reshape([textdim, 512])
        if zero_pad:
            cc0 = numpy.zeros((cc.shape[0]+1, cc.shape[1])).astype('float32')
            cc0[:-1,:] = cc
            tt0 = numpy.zeros((tt.shape[0]+1, tt.shape[1])).astype('float32')
            tt0[:-1,:] = tt.astype(numpy.float32)
        else:
            cc0 = cc
            tt0 = tt.astype(numpy.float32)
        return create_sample(tparams, f_init, f_next, cc0, tt0, options, trng, k, normalize)

    ds = datasets.strip().split(',')

    # send all the features for the various datasets
    for dd in ds:
        if dd == 'dev':
            bar = Bar('Development Set...', max=len(valid[1]))
            caps = []
            for i in range(len(valid[1])):
                sample = _send_job(valid[1][i], valid[2][i])
                cap = _seqs2words(sample)
                caps.append(cap)
                with open(saveto+'_status.json', 'w') as f:
                    json.dump({'current': i, 'total': len(valid[1])}, f)
                bar.next()
            bar.finish()
            with open(saveto, 'w') as f:
                print >>f, '\n'.join(caps)
            print 'Done'
        if dd == 'test':
            bar = Bar('Test Set...', max=len(test[1]))
            caps = []
            for i in range(len(test[1])):
                sample = _send_job(test[1][i], test[2][i])
                cap = _seqs2words(sample)
                caps.append(cap)
                with open(saveto+'_status.json', 'w') as f:
                    json.dump({'current': i, 'total': len(test[1])}, f)
                bar.next()
            bar.finish()
            with open(saveto, 'w') as f:
                print >>f, '\n'.join(caps)
            print 'Done'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, default=1)
    parser.add_argument('-sampling', action="store_true", default=False) # this only matters for hard attention
    parser.add_argument('-n', action="store_true", default=False)
    parser.add_argument('-z', action="store_true", default=False)
    parser.add_argument('-d', type=str, default='dev')
    parser.add_argument('-pkl_name', type=str, default=None, help="name of pickle file (without the .pkl)")
    parser.add_argument('dataset', type=str)
    parser.add_argument('model', type=str)
    parser.add_argument('saveto', type=str)

    args = parser.parse_args()
    main(args.model, args.saveto, args.dataset, k=args.k, zero_pad=args.z, pkl_name=args.pkl_name, normalize=args.n, datasets=args.d, sampling=args.sampling)