import argparse
import cPickle as pkl
import sys

# Status monitor
from monitor import Monitor

from IPython import embed

def main(args):
    monitor = Monitor('{}/{}_status.json'.format(args['out_dir'].rstrip('/'), args["model"]))

    try:
        if args['type'] == 'normal':
            from capgen import train

            _, validerr, _ = train(out_dir=args['out_dir'].rstrip('/'),
                                   data_dir=args['data_dir'].rstrip('/'),
                                   saveto=args["model"],
                                   attn_type='deterministic',
                                   reload_=args['reload'],
                                   dim_word=512,
                                   ctx_dim=512,
                                   dim=1800,
                                   n_layers_att=2,
                                   n_layers_out=1,
                                   n_layers_lstm=1,
                                   n_layers_init=2,
                                   n_words=10000,
                                   lstm_encoder=False,
                                   decay_c=0.,
                                   alpha_c=1.,
                                   prev2out=True,
                                   ctx2out=True,
                                   lrate=0.01,
                                   optimizer='adam',
                                   selector=True,
                                   patience=10,
                                   maxlen=100,
                                   batch_size=64,
                                   valid_batch_size=64,
                                   validFreq=2000,
                                   dispFreq=1,
                                   saveFreq=1000,
                                   sampleFreq=250,
                                   dataset="coco",
                                   use_dropout=True,
                                   use_dropout_lstm=False,
                                   save_per_epoch=False,
                                   monitor=monitor)
            print "Final cost: {:.2f}".format(validerr.mean())
        elif args['type'] == 't_attn':
            from capgen_text import train

            out_dir = args['out_dir'].rstrip('/')
            saveto  = args['model']
            _, validerr, _ = train(out_dir=out_dir,
                                   data_dir=args['data_dir'].rstrip('/'),
                                   saveto=saveto,
                                   attn_type='deterministic',
                                   reload_=args['reload'],
                                   dim_word=512,
                                   ctx_dim=512,
                                   tex_dim=args['tex_dim'],
                                   dim=1800,
                                   n_layers_att=2,
                                   n_layers_out=1,
                                   n_layers_lstm=1,
                                   n_layers_init=2,
                                   n_words=10000,
                                   lstm_encoder=False,
                                   lstm_encoder_context=args['lenc'],
                                   decay_c=0.,
                                   alpha_c=1.,
                                   prev2out=True,
                                   ctx2out=True,
                                   tex2out=True,
                                   lrate=0.01,
                                   optimizer='adam',
                                   selector=True,
                                   patience=10,
                                   maxlen=100,
                                   batch_size=32,
                                   valid_batch_size=32,
                                   validFreq=2000,
                                   dispFreq=1,
                                   saveFreq=1000,
                                   sampleFreq=250,
                                   dataset="coco",
                                   use_dropout=True,
                                   use_dropout_lstm=False,
                                   save_per_epoch=False,
                                   monitor=monitor)
            print "Final cost: {:.2f}".format(validerr.mean())

            # Store data preprocessing type in the options file
            with open('{}/{}.pkl'.format(out_dir, saveto)) as f_opts:
                opts = pkl.load(f_opts)
                opts['preproc_type'] = args['preproc_type']
                preproc_params = {}
                for param in args['preproc_params'].split(','):
                    if param:
                        key, value = param.split('=')
                        if value.isdigit():
                            value = int(value)
                        preproc_params[key] = value
                opts['preproc_params'] = preproc_params
                pkl.dump(opts, f_opts)
    except (KeyboardInterrupt, SystemExit):
        print 'Interrupted!'
        monitor.error_message = 'Interrupted!'
        monitor.status = 12
    except Exception, e:
        print 'Unexpected error!'
        monitor.error_message = str(e)
        monitor.status = 12
        raise e

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='data directory')
    parser.add_argument('out_dir', help='output directory')
    parser.add_argument('model', help='model filename (*.npz)')
    parser.add_argument("--attn_type",  default="deterministic", help="type of attention mechanism", choices=['deterministic', 'stochastic'])
    parser.add_argument('--type', default='normal', choices=['normal', 't_attn'])
    parser.add_argument('--reload', '-r', help='reload model', action='store_true')
    parser.add_argument('--lenc', help='use lstm context encoding', action='store_true')
    parser.add_argument('--tex_dim', help='dimensionality of context', type=int, default=512)
    parser.add_argument('--preproc_type', help='preprocessing type', choices=['raw','tfidf','w2v','w2vtfidf'])
    parser.add_argument('--preproc_params', help='preprocessing params', type=str)
    args = parser.parse_args()
    main(vars(args))