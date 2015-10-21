import json
import nltk
import numpy as np
from progress.bar import Bar

from IPython import embed

def main():
	d = json.load(open('captions_val2014.json', 'r'))

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

	with open('splits/coco_val.txt', 'r') as f:
		images = f.read().split()

	refs = []
	for im in images:
		refs.append('<>'.join(id2cap[im2id[im]]))

	with open('first_run.references.txt', 'w') as f:
		print >>f, '\n'.join(refs)

if __name__ == '__main__':
	main()