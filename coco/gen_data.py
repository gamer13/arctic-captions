import os
import caffe
import numpy as np
import json
import pandas as pd
import scipy
import time
import h5py
from sklearn.cross_validation import train_test_split
from progress.bar import Bar
import cPickle
import argparse

from IPython import embed

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate data for the MSCOCO dataset')

    parser.add_argument('--proto', dest='prototxt', help='Deploy prototxt file for CNN', type=str)
    parser.add_argument('--model', dest='model', help='Caffemodel file for CNN', type=str)
    parser.add_argument('--mean', dest='mean', help='Mean file for CNN', type=str)
    parser.add_argument('--imgs', dest='image_dir', help='Input image directory', type=str)
    parser.add_argument('--cin', dest='caps_in', help='Captions JSON data file', type=str)
    parser.add_argument('-o', dest='out_dir', help='Output directory', type=str)
    parser.add_argument('--cdim', dest='cnn_dim', default='224,224', type=str, help='CNN input dimensions: width,height')
    parser.add_argument('-b', dest='batch_size', default=50, type=int, help='CNN batch size')
    parser.add_argument('-s', dest='seed', type=int, default=1234, help='Random seed')

    args = parser.parse_args()

    print 'Settings: ', args

    cnn_prototxt    = args.prototxt
    cnn_model       = args.model
    cnn_mean        = args.mean
    image_path      = args.image_dir
    captions_in     = args.caps_in
    captions_out    = args.out_dir
    cnn_in_width, cnn_in_height = map(int, args.cnn_dim.split(','))
    batch_size      = args.batch_size
    seed            = args.seed

    # Use GPU processing, because faster = better
    caffe.set_mode_gpu()

    # Construct CNN from input files
    cnn         = caffe.Net(cnn_prototxt, cnn_model, caffe.TEST)
    # Configure transformer for CNN input images
    transformer = caffe.io.Transformer({'data': cnn.blobs['data'].data.shape})
    transformer.set_mean('data', np.load(cnn_mean).mean(1).mean(1))
    transformer.set_transpose('data', (2,0,1))
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2,1,0))

    # Setup CNN such that it can process n_samples = batch_size
    cnn.blobs['data'].reshape(batch_size, 3, cnn_in_width, cnn_in_height)

    ## Process sentences
    json_data = json.load(open(captions_in, 'r')) # Load captions
    image_dataset = json_data['images']
    print 'Done loading json file'

    ## Read JSON data
    bar = Bar('Reading...', max=len(image_dataset), suffix='%(percent)d%%')
    rows = []
    ignore_images_list = ['COCO_train2014_000000167126.jpg'] # Hacky, but necessary
    use_splits_list = ['train']
    for image in image_dataset:
        for sentence in image['sentences']:
            if image['filename'] not in ignore_images_list and \
               image['split'] in use_splits_list:
                rows.append({
                    'image': image['filename'], 
                    'caption': sentence['raw']
                })
        bar.next()
    bar.finish()

    # Convert to Pandas dataframe format for easy processing
    # Columns: image | caption
    captions_df = pd.DataFrame(rows)
    print 'Loaded {} captions for {} unique images'.format(captions_df.shape[0], captions_df['image'].unique().shape[0])

    captions = captions_df['caption'].values # Get all values in 'caption' column

    # Count the total number of occurrences of each word
    bar = Bar('Counting...', max=len(captions), suffix='%(percent)d%%')
    vocabulary = {}
    for caption in captions:
        for token in caption.split():
            vocabulary[token.lower()] = vocabulary.get(token.lower(), 0) + 1
        bar.next()
    bar.finish()
    # Add 2 to indexes for <eos> and UNK tokens
    dictionary_series = pd.Series(vocabulary.values(), index=vocabulary.keys()) + 2
    # Change back to dict
    dictionary = dictionary_series.to_dict()

    # Write dictionary
    with open('{}/dictionary.pkl'.format(captions_out), 'wb') as f:
        cPickle.dump(dictionary, f)

    # Get all unique images
    images = pd.Series(captions_df['image'].unique())
    # Dict: image -> image id
    image_id_dict = pd.Series(np.array(images.index), index=images)
    # Dict: image ids of captions
    caption_image_id = captions_df['image'].map(lambda x: image_id_dict[x])
    # Tuples of (caption,image id)
    cap = zip(captions, caption_image_id)
    cap_df = pd.DataFrame(cap)

    # Process images
    all_idxs = range(len(images))
    # Split into train, test and dev sets
    train_idxs, test = train_test_split(all_idxs, test_size=0.3, random_state=seed)
    test_idxs, dev_idxs = train_test_split(test, test_size=0.4, random_state=seed)

    try:
        splits = {'train': train_idxs, 'test': test_idxs, 'dev': dev_idxs}
        for split, split_idxs in splits.items():
            print 'Processing split {}'.format(split)

            cnn.blobs['data'].reshape(batch_size, 3, cnn_in_width, cnn_in_height)

            # Get images for the split
            images_split = images[split_idxs].sort_index()
            files_split  = images_split.values
            image_id_dict_split = image_id_dict[split_idxs].sort_index().values
            caption_image_id_split = cap_df[cap_df[1].isin(split_idxs)][1].values
            captions_split  = cap_df[cap_df[1].isin(split_idxs)][0].values

            images_lookup = np.zeros((len(images),), dtype=int)
            images_lookup.fill(-1)
            for idx, image_id in enumerate(images_split.index):
                images_lookup[image_id] = idx

            bar = Bar('Processing...', max=len(images_split))
            cache_split = h5py.File('{}/coco_{}.h5'.format(captions_out, split), 'w')
            g_imgs   = cache_split.create_group('images')
            g_caps   = cache_split.create_group('captions')

            dfeats   = g_imgs.create_dataset('feats', (len(images_split), 512*14*14), dtype=np.float32)
            dimgs    = g_imgs.create_dataset('imgs', (len(images_split),), dtype=h5py.special_dtype(vlen=bytes))
            dlookup  = g_imgs.create_dataset('lookup', (len(images),), dtype=int)
            dcaps    = g_caps.create_dataset('caps', (len(captions_split),), dtype=h5py.special_dtype(vlen=unicode))
            dimg_ids = g_caps.create_dataset('img_ids', (len(captions_split),), dtype=int)

            dlookup[...]  = images_lookup
            dcaps[...]    = captions_split
            dimg_ids[...] = caption_image_id_split

            average_per_batch = []
            for i in xrange(0, batch_size, batch_size):
                bar.goto(i)

                time_b = time.clock()

                # Grab files for batch
                image_files = files_split[i : i + batch_size]

                # Create CNN input with preprocessing pipeline
                cnn_in = np.zeros((batch_size, 3, 224, 224), dtype=np.float32)
                for img_idx, img in enumerate(image_files):
                    cnn_in[img_idx, :] = transformer.preprocess('data', caffe.io.load_image('{}/{}/{}'.format(image_path, 'train', img)))

                # Get features output
                out = cnn.forward_all(blobs=['conv5_4'], **{'data': cnn_in})
                flatten = np.array(map(lambda x: x.flatten(), out['conv5_4']))
                dfeats[i : i + batch_size] = flatten[0:len(image_files)]
                dimgs[i: i + batch_size]   = image_files

                average_per_batch.append(time.clock() - time_b)

                if i % 1000 == 0 and i != 0:
                    print '\n{}s per batch'.format(np.mean(average_per_batch))
                    average_per_batch = [] # Empty it again
            bar.finish()
    except:
        embed()
        raise
            