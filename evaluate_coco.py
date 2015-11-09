"""
Example execution script. The dataset parameter can
be modified to coco/flickr30k/flickr8k
"""
import argparse
import sys

#from capgen import train
from capgen_text import train

# Status monitor
from monitor import Monitor

def main(params):
    # see documentation in capgen.py for more details on hyperparams
    monitor = Monitor('{}/{}_status.json'.format(params['out_dir'].rstrip('/'), params["model"]))

    #try:
    _, validerr, _ = train(out_dir=params['out_dir'].rstrip('/'),
                           saveto=params["model"],
                           attn_type=params["attn_type"],
                           reload_=params["reload"],
                           dim_word=params["dim_word"],
                           ctx_dim=params["ctx_dim"],
                           tex_dim=params["tex_dim"],
                           dim=params["dim"],
                           n_layers_att=params["n_layers_att"],
                           n_layers_out=params["n_layers_out"],
                           n_layers_lstm=params["n_layers_lstm"],
                           n_layers_init=params["n_layers_init"],
                           n_words=params["n_words"],
                           lstm_encoder=params["lstm_encoder"],
                           lstm_encoder_context=params["lstm_encoder_context"],
                           decay_c=params["decay_c"],
                           alpha_c=params["alpha_c"],
                           prev2out=params["prev2out"],
                           ctx2out=params["ctx2out"],
                           tex2out=params["tex2out"],
                           lrate=params["lr"],
                           optimizer=params["optimizer"],
                           selector=params["selector"],
                           patience=10,
                           maxlen=100,
                           batch_size=32,
                           valid_batch_size=32,
                           validFreq=2000,
                           dispFreq=1,
                           saveFreq=1000,
                           sampleFreq=250,
                           dataset="coco",
                           use_dropout=params["use_dropout"],
                           use_dropout_lstm=params["use_dropout_lstm"],
                           save_per_epoch=params["save_per_epoch"],
                           monitor=monitor)
    print "Final cost: {:.2f}".format(validerr.mean())
    #except (KeyboardInterrupt, SystemExit):
    #    print 'Interrupted!'
    #    monitor.error_message = 'Interrupted!'
    #    monitor.status = 12
    #except Exception, e:
    #    print 'Unexpected error!'
    #    monitor.error_message = str(e)
    #    monitor.status = 12
    #    raise e


if __name__ == "__main__":
    # These defaults should more or less reproduce the soft alignment model for the MS COCO dataset
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir', help='output directory')
    parser.add_argument('model', help='model filename (*.npz)')
    parser.add_argument("--attn_type",  default="deterministic", help="type of attention mechanism", choices=['deterministic', 'stochastic'])
    parser.add_argument('--dim_word', default=512, help='word vector dimensionality')
    parser.add_argument('--ctx_dim', default=512, help='context vector dimensionality')
    parser.add_argument('--tex_dim', default=512, help='textual context vector dimensionality')
    parser.add_argument('--dim', default=1800, help='the number of LSTM units')
    parser.add_argument('--n_layers_att', default=2, help='number of layers used to compute the attention weights')
    parser.add_argument('--n_layers_out', default=1, help='number of layers used to compute logit')
    parser.add_argument('--n_layers_lstm', default=1, help='number of lstm layers')
    parser.add_argument('--n_layers_init', default=2, help='number of layers to initialize LSTM at time 0')
    parser.add_argument('--n_words', default=10000, help='vocab size')
    parser.add_argument('--lstm_encoder', default=False, help='if True, run bidirectional LSTM on input units')
    parser.add_argument('--lstm_encoder_context', default=False)
    parser.add_argument('--decay_c', default=0., help='weight decay coefficient')
    parser.add_argument('--alpha_c', default=1., help='doubly stochastic coefficient')
    parser.add_argument('--prev2out', default=True, help='Feed previous word into logit')
    parser.add_argument('--ctx2out', default=True, help='Feed attention weighted ctx into logit')
    parser.add_argument('--tex2out', default=True, help='Feed attention weighted tex into logit')
    parser.add_argument('--lr', default=0.01, help='used only for SGD')
    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--selector', default=True, help='selector (see paper)')
    parser.add_argument('--use_dropout', default=True, help='setting this true turns on dropout at various points')
    parser.add_argument('--use_dropout_lstm', default=False, help='dropout on lstm gates')
    parser.add_argument('--save_per_epoch', default=False, help='this saves down the model every epoch')
    parser.add_argument('--reload', default=False)
    args = parser.parse_args()

    main(vars(args))