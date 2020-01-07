#!/usr/bin/env python

import os
import shutil

def rename_images_by_number(image_dir, prefix):
    for idx, image_fp in enumerate(os.listdir(image_dir)):
        new_fp = prefix + '{:05d}.jpg'.format(idx+1)
        shutil.move(os.path.join(image_dir, image_fp),
                    os.path.join(image_dir, new_fp))
        
if __name__ == '__main__':
    image_dir = 'data/stock-images/male-images'
    # rename_images_by_number(image_dir, 'M')
    
    image_dir = 'data/stock-images/female-images'
    #rename_images_by_number(image_dir, 'F')
    
