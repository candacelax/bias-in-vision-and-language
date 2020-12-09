# coding=utf-8
# Copyleft 2019 Project LXRT

import sys
import csv
import base64
import time
import numpy as np
import lmdb
import pickle
import copy

csv.field_size_limit(sys.maxsize)
FIELDNAMES_COCO = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
                "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
FIELDNAMES_GOOGLE = ['img_id', 'img_w','img_h','num_boxes', 'boxes', 'features', 'cls_prob']


def load_obj_tsv(fname, topk=None):
    """Load object features from tsv file.

    :param fname: The path to the tsv file.
    :param topk: Only load features for top K images (lines) in the tsv file.
        Will load all the features if topk is either -1 or None.
    :return: A list of image object features where each feature is a dict.
        See FILENAMES above for the keys in the feature dict.
    """
    fieldnames = FIELDNAMES_COCO if 'coco' in fname else FIELDNAMES_GOOGLE
    data = []
    start_time = time.time()
    print("Start to load Faster-RCNN detected objects from %s" % fname)
    if 'lmdb' in fname:
        env = lmdb.open(fname, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)

        with env.begin(write=False) as txn:
            image_ids = pickle.loads(txn.get('keys'.encode()))
            features = format_lmdb(env, image_ids)
            return features

    with open(fname) as f:
        reader = csv.DictReader(f, fieldnames, delimiter="\t")
        for i, item in enumerate(reader):
            for key in ['img_h', 'img_w', 'num_boxes']:
                item[key] = int(item[key])
            
            boxes = item['num_boxes']
            decode_config = [
                ('objects_id', (boxes, ), np.int64),
                ('objects_conf', (boxes, ), np.float32),
                ('attrs_id', (boxes, ), np.int64),
                ('attrs_conf', (boxes, ), np.float32),
                ('boxes', (boxes, 4), np.float32),
                ('features', (boxes, -1), np.float32),
            ]
            for key, shape, dtype in decode_config:
                if not key in item:
                    item[key] = None
                    continue
                item[key] = np.frombuffer(base64.b64decode(item[key]), dtype=dtype)
                item[key] = item[key].reshape(shape)
                item[key].setflags(write=False)

            data.append(item)
            if topk is not None and len(data) == topk:
                break
    elapsed_time = time.time() - start_time
    print("Loaded %d images in file %s in %d seconds." % (len(data), fname, elapsed_time))
    return data


""" additionally returns object class
"""
def format_lmdb(env, image_ids):
    # Read chunk from file everytime if not loaded in memory.    
    with env.begin(write=False) as txn:
        data = []
        for img_id in image_ids:
            item = pickle.loads(txn.get(img_id))
            image_id = item['image_id']
            image_h = int(item['image_h'])
            image_w = int(item['image_w'])
            num_boxes = int(item['num_boxes'])

            features = np.frombuffer(base64.b64decode(item["features"]), dtype=np.float32)\
                            .reshape(num_boxes, -1)
            boxes = np.frombuffer(base64.b64decode(item['boxes']), dtype=np.float32)\
                        .reshape(num_boxes, 4)
            cls_indices = np.frombuffer(base64.b64decode(item['classes'])).astype(np.int32)
                
            
            #image_location = np.zeros((boxes.shape[0], 5), dtype=np.float32)
            #image_location[:,:4] = boxes
            #image_location[:,4] = (image_location[:,3] - image_location[:,1]) * (image_location[:,2] - image_location[:,0]) / (float(image_w) * float(image_h))

            #image_location_ori = copy.deepcopy(image_location)
            #image_location[:,0] = image_location[:,0] / float(image_w)
            #image_location[:,1] = image_location[:,1] / float(image_h)
            #image_location[:,2] = image_location[:,2] / float(image_w)
            #image_location[:,3] = image_location[:,3] / float(image_h)

            #g_location = np.array([0,0,1,1,1])
            #image_location = np.concatenate([np.expand_dims(g_location, axis=0), image_location], axis=0)

            #g_location_ori = np.array([0,0,image_w,image_h,image_w*image_h])
            #image_location_ori = np.concatenate([np.expand_dims(g_location_ori, axis=0), image_location_ori], axis=0)
            data.append(
                {
                    'img_id' : image_id,
                    'img_h' : image_h,
                    'img_w' : image_w,
                    'features' : features,
                    'num_boxes': num_boxes,
                    'boxes' : boxes,
                    'objects_id' : cls_indices
                })
            assert num_boxes == len(boxes) == len(features), f'{num_boxes} {len(boxes)} {len(features)}'
        return data
