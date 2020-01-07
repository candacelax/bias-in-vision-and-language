#!/usr/bin/env python

# python eval_pretrained.py -folder ../logs -config ../configs/nlvr2/coco-pre-train.json

# TODO make sure nothing is being masked during eval
# TODO make sure all dropout is zero
# TODO check for image overlap
# TODO text labels for enc
# TODO pooled versus sequence_output
# TODO check indexing of cosine sim
# TODO check all ID1/ID2 indexing for typos

# ERROR: allennlp 0.8.0 requires awscli>=1.11.91, which is not installed.
# ERROR: allennlp 0.8.0 requires flask-cors==3.0.7, which is not installed.
# ERROR: allennlp 0.8.0 requires moto==1.3.4, which is not installed.
# ERROR: allennlp 0.8.0 requires pytorch-pretrained-bert==0.3.0, which is not installed.
# ERROR: allennlp 0.8.0 requires responses>=0.7, which is not installed.
# ERROR: thinc 6.12.1 has requirement msgpack-numpy<0.4.4, but you'll have msgpack-numpy 0.4.4.3 which is incompatible.
# ERROR: spacy 2.0.16 has requirement msgpack-numpy<0.4.4, but you'll have msgpack-numpy 0.4.4.3 which is incompatible.
# ERROR: spacy 2.0.16 has requirement regex==2018.01.10, but you'll have regex 2019.12.9 which is incompatible.


import argparse
import os
import re
import logging as log
from time import localtime
import commentjson
import yaml

import torch
from attrdict import AttrDict
from scripts.loader import load_model, load_data
from scripts.encoder import EncoderWrapper
from scripts import weat_images as weat


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/visualbert_coco_pre.yaml',
                        help='TODO')
    return parser.parse_args()

if __name__ == '__main__':
    # general params
    config_fp = parse_args().config
    with open(config_fp) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    params = AttrDict(config)

    # model-specific params
    if params.get('model_config'):
        with open(params.model_config) as f:
            model_config = AttrDict(commentjson.load(f))
            #args = ModelWrapper.read_and_insert_args(model_config, params.model_config)
            params.update(model_config)
    
    # set up log
    t = localtime()
    timestamp = f'{t.tm_mon}-{t.tm_mday}-{t.tm_year}_' +\
                f'{t.tm_hour:2d}:{t.tm_min:2d}:{t.tm_sec:2d}'
    log_fpath = os.path.join(params.log_dir, timestamp)
    log.getLogger().addHandler(log.FileHandler(log_fpath))
    log.info(f'Params: {params}')


    # Load model
    # from visualbert.dataloaders.vcr import VCRLoader
    # from visualbert.dataloaders.coco_dataset import COCODataset
    # coco_params = deepcopy(params)
    # coco_params['text_only'] = True
    # train,val,test = COCODataset.splits(params)
    # loader_params = {'batch_size': params.train_batch_size // params.num_gpus,
    #                  'num_gpus': params.num_gpus,
    #                  'num_workers': params.num_workers}
    # train_loader = VCRLoader.from_dataset(train, **loader_params)

    #model = load_model(params)
    #encoder = EncoderWrapper(model)

    # Load tests
    log.info('Starting test loader')
    if isinstance(params.tests, tuple) or isinstance(params.tests, list):
        test_fpaths = params.tests
    else:
        test_fpaths = [os.path.join(params.tests, fp) for fp in \
                       os.listdir(params.tests)]

    for tfp in test_fpaths:
        # load test
        log.info(f'Loading {tfp}')
        dataloaders = load_data(params, fp=tfp)

        # targets w/ corresponding images
        encoded_targ1 = encoder.encode(dataloaders['targ1'])
        encoded_targ2 = encoder.encode(dataloaders['targ2'])
                
        # attr1, male and female images
        encoded_attr1_male = encoder.encode(dataloaders['attr1_male_images'])
        encoded_attr1_female = encoder.encode(dataloaders['attr1_female_images'])

        # attr2, male and female images
        encoded_attr2_male = encoder.encode(dataloaders['attr2_male_images'])
        encoded_attr2_female = encoder.encode(dataloaders['attr2_female_images'])

        encodings = {'targ1' : {'encs' : encoded_targ1,
                                'category' : dataloaders['targ1'].dataset.category},
                     'targ2' : {'encs' : encoded_targ2,
                                'category' : dataloaders['targ2'].dataset.category},
                     'attr1_ID1' : {'encs' : encoded_attr1_male,
                                    'category' : dataloaders['attr1_male_images'].dataset.category},
                     'attr1_ID2' : {'encs' : encoded_attr1_female,
                                    'category' : dataloaders['attr1_female_images'].dataset.category},
                     'attr2_ID1' : {'encs' : encoded_attr2_male,
                                    'category' : dataloaders['attr2_male_images'].dataset.category},
                     'attr2_ID2' : {'encs' : encoded_attr2_female,
                                    'category' : dataloaders['attr2_female_images'].dataset.category},
        }
                     
        
        weat.run_test(encodings, n_samples=params.num_samples)

        
