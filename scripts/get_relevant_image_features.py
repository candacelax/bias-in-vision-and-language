#!/usr/bin/env python

import torch
from scripts.utils import load_json
import os

if __name__ == '__main__':
    full_chunk_path = 'visualbert/image-features/coco_val_image_features.th'
    selected_chunk_path = 'visualbert/image-features/coco_bias_image_features.th'
    chunk = torch.load(full_chunk_path)

    image_prefix = 'COCO_val2014_'
    tests_path = 'tests/tests-images'

    for fp in os.listdir(tests_path):
        data = load_json(os.path.join(tests_path, fp))
        examples = data['attr1']['examples']['male_images'] + \
                   data['attr1']['examples']['female_images'] + \
                   data['attr2']['examples']['male_images'] + \
                   data['attr2']['examples']['female_images']


        filtered = {}
        image_lists = set()
        for e in examples:
            if isinstance(e, list):
                for image_fp in e[1:]:
                    image_fp = f'{image_prefix}{image_fp}.jpg'
                    if image_fp in filtered:
                        continue

                    filtered[image_fp] = chunk[image_fp]
        torch.save(filtered, selected_chunk_path)
            
