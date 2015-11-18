import json
import nltk
import numpy as np
import argparse
from progress.bar import Bar

from IPython import embed

def main(args):
	d = json.load(open(args.c, 'r'))

	np.random.seed(1234)

	im2id  = {}
	id2cap = {}

	print 'img 2 id....'
	for im in d['images']:
		im2id[im['file_name']] = im['id']

	bar = Bar('id 2 cap...', max=len(d['annotations']))
	for ann in d['annotations']:
		cap = nltk.word_tokenize(ann['caption'])
		cap = ' '.join(cap).lower()
		if ann['image_id'] in id2cap:
			id2cap[ann['image_id']].append(cap)
		else:
			id2cap[ann['image_id']] = [cap]
		bar.next()
	bar.finish()

	with open(args.s, 'r') as f:
		images = f.read().split()

	refs = []
	for im in images:
		refs.append('<>'.join(id2cap[im2id[im]]))

	with open(args.saveto, 'w') as f:
		print >>f, '\n'.join(refs)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-c', help='Validation captions file', type=str)
	parser.add_argument('-s', help='Validation split file', type=str)
	parser.add_argument('saveto', help='Output file', type=str)
	args = parser.parse_args()

	main(args)