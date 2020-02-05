import h5py
import os
from os import path
import pdb
import numpy as np
import json
import sys
FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
import csv
import base64
import pickle
import lmdb # install lmdb by "pip install lmdb"
import re

csv.field_size_limit(sys.maxsize)

count = 0
prefix = sys.argv[1]
save_path = re.sub('.tsv', '.lmdb', prefix)
env = lmdb.open(save_path, map_size=1099511627776)

infiles = []
dirname=path.dirname(prefix)

for f in os.listdir(dirname):
    if re.match(prefix+'.[0-9]+', path.join(dirname, f)):
        infiles.append(path.join(dirname,f))

id_list = []
with env.begin(write=True) as txn:
    for infile in infiles:
        with open(infile) as tsv_in_file:
            reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
            for item in reader:
                img_id = str(item['image_id']).encode()
                id_list.append(img_id)
                txn.put(img_id, pickle.dumps(item))
                if count % 1000 == 0:
                    print(count)
                count += 1
    txn.put('keys'.encode(), pickle.dumps(id_list))

print(count)
