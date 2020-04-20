#!/usr/bin/env python

import os
import re

import logging as log
from time import localtime
import csv

import torch
from scripts import utils
from scripts.loader import load_model, load_data
from scripts.encoder import EncoderWrapper
from scripts.weat import weat_images_union as weat_union
from scripts.weat import weat_images_targ_specific as weat_specific
from scripts.weat import weat_images_intra_targ as weat_intra
from scripts.weat.general_vals import get_general_vals # TODO rename

if __name__ == '__main__':
    # Load params and set up logging
    params = utils.load_params()
    t = localtime()
    timestamp = f'{t.tm_mon}-{t.tm_mday}-{t.tm_year}_' +\
                f'{t.tm_hour:02d}:{t.tm_min:02d}:{t.tm_sec:02d}'
    log_fpath = os.path.join(params.log_dir, timestamp)
    log.getLogger().addHandler(log.FileHandler(log_fpath))
    log.info(f'Params: {params}')
    
    os.mkdir(f'results/{timestamp}')
    f_exp1 = open(f'results/{timestamp}/exp1.csv', 'w')
    f_exp2 = open(f'results/{timestamp}/exp2.csv', 'w')
    f_exp3 = open(f'results/{timestamp}/exp3.csv', 'w')
    f_gen = open(f'results/{timestamp}/general.csv', 'w')

    writer_exp1 = csv.writer(f_exp1, delimiter=',')
    writer_exp2 = csv.writer(f_exp2, delimiter=',')
    writer_exp3 = csv.writer(f_exp3, delimiter=',')
    writer_gen = csv.writer(f_gen, delimiter=',')

    writer_exp1.writerow(['test_name', 'esize', 'pval'])
    writer_exp2.writerow(['test_name', 'esize', 'pval'])
    writer_exp3.writerow(['test_name', 'esize_x', 'pval_x', 'esize_y', 'pval_y'])
    writer_gen.writerow(['test_name', 'X_AXonAY', 'X_BXonBY',
                         'Y_AXonAY', 'Y_BXonBY', 'X_AonB', 'Y_AonB',
                         'X_ABXonABY', 'Y_ABXonABY'])

    #--- Load model
    model = load_model(params)
    encoder = EncoderWrapper(model)
        
    #--- Load tests
    log.info('Starting test loader')
    for vals in zip(params.tests, params.features):
        bias_test_fp, chunk_fp = vals

        # load test
        log.info(f'Loading {bias_test_fp} {chunk_fp}')
        params.chunk_path = chunk_fp
        params.features_fpath = chunk_fp
        dataloaders = load_data(params, bias_test_fp)
        log.info('Total number of unique images: {}'.format(
            dataloaders['targ_X'].dataset.getNumUniqueImages() + \
            dataloaders['targ_Y'].dataset.getNumUniqueImages() + \
            dataloaders['attr_A_X'].dataset.getNumUniqueImages() + \
            dataloaders['attr_A_Y'].dataset.getNumUniqueImages() + \
            dataloaders['attr_B_X'].dataset.getNumUniqueImages() + \
            dataloaders['attr_B_Y'].dataset.getNumUniqueImages()))
        
        test_type = 'sent' if re.match('.*sent.*', bias_test_fp) else 'word'
        test_name = re.sub('sent-|.jsonl', '', os.path.basename(bias_test_fp))
        test_name = re.sub('one_sentence|one_word', '', test_name)

        # targets w/ corresponding images
        encoded_X, contextual_X = encoder.encode(dataloaders['targ_X'])
        encoded_Y, contextual_Y = encoder.encode(dataloaders['targ_Y'])
                
        # attribute A with images corresponding to targets X and Y
        encoded_attr_A_X, contextual_A_X = encoder.encode(dataloaders['attr_A_X'])
        encoded_attr_A_Y, contextual_A_Y = encoder.encode(dataloaders['attr_A_Y'])

        # attribute B with images corresponding to targets X and Y
        encoded_attr_B_X, contextual_B_X = encoder.encode(dataloaders['attr_B_X'])
        encoded_attr_B_Y, contextual_B_Y = encoder.encode(dataloaders['attr_B_Y'])

        encodings = {'targ_X' : {'encs' : encoded_X,
                                 'category' : dataloaders['targ_X'].dataset.category},
                     'targ_Y' : {'encs' : encoded_Y,
                                 'category' : dataloaders['targ_Y'].dataset.category},
                     'attr_A_X' : {'encs' : encoded_attr_A_X,
                                   'category' : dataloaders['attr_A_X'].dataset.category},
                     'attr_A_Y' : {'encs' : encoded_attr_A_Y,
                                   'category' : dataloaders['attr_A_Y'].dataset.category},
                     'attr_B_X' : {'encs' : encoded_attr_B_X,
                                   'category' : dataloaders['attr_B_X'].dataset.category},
                     'attr_B_Y' : {'encs' : encoded_attr_B_Y,
                                   'category' : dataloaders['attr_B_Y'].dataset.category}
        }

        log.info('Running experiment 1: union across attribute A_{X} and A_{Y}')
        esize, pval = weat_union.run_test(encodings, n_samples=params.num_samples)
        writer_exp1.writerow([f'{test_name}:{test_type}', esize, pval])

        log.info('\n\n')
        log.info('Running experiment 2: corresponding attributes s(X, A_{X}, B_{X}) and s(Y, A_{Y}, B_{Y})')
        esize, pval = weat_specific.run_test(encodings, n_samples=params.num_samples)
        writer_exp2.writerow([f'{test_name}:{test_type}', esize, pval])  
        
        log.info('\n\n')
        log.info('Running experiment 3: intra-target across target-specific attributes')
        esize_x, pval_x, esize_y, pval_y = weat_intra.run_test(encodings, n_samples=params.num_samples)
        writer_exp3.writerow([f'{test_name}:{test_type}', esize_x, pval_x, esize_y, pval_y])
        
        log.info('\n\n')
        log.info('Computing general cosine similarities between cells')
        vals = get_general_vals(encodings, n_samples=params.num_samples)
        writer_gen.writerow([f'{test_name}:{test_type}', vals['X_AXonAY'], vals['X_BXonBY'],
                             vals['Y_AXonAY'], vals['Y_BXonBY'], vals['X_AonB'],
                             vals['Y_AonB'], vals['X_ABXonABY'], vals['Y_ABXonABY']])

        if re.match('.*sent.', bias_test_fp):
            # run over contextual encodings
            test_type = 'contextual'
            encodings = {'targ_X' : {'encs' : contextual_X,
                                     'category' : dataloaders['targ_X'].dataset.category},
                         'targ_Y' : {'encs' : contextual_Y,
                                     'category' : dataloaders['targ_Y'].dataset.category},
                         'attr_A_X' : {'encs' : contextual_A_X,
                                       'category' : dataloaders['attr_A_X'].dataset.category},
                         'attr_A_Y' : {'encs' : contextual_A_Y,
                                       'category' : dataloaders['attr_A_Y'].dataset.category},
                         'attr_B_X' : {'encs' : contextual_B_X,
                                       'category' : dataloaders['attr_B_X'].dataset.category},
                         'attr_B_Y' : {'encs' : contextual_B_Y,
                                       'category' : dataloaders['attr_B_Y'].dataset.category}
            }

            log.info('Running experiment 1 (contextual): union across attribute A_{X} and A_{Y}')
            esize, pval = weat_union.run_test(encodings, n_samples=params.num_samples)
            writer_exp1.writerow([f'{test_name}:{test_type}', esize, pval])
            
            log.info('Running experiment 2 (contextual): corresponding attributes s(X, A_{X}, B_{X}) and s(Y, A_{Y}, B_{Y})')
            esize, pval = weat_specific.run_test(encodings, n_samples=params.num_samples)
            writer_exp2.writerow([f'{test_name}:{test_type}', esize, pval])

            log.info('\n\n')
            log.info('Running experiment 3 (contextual): intra-target across target-specific attributes')
            esize_x, pval_x, esize_y, pval_y = weat_intra.run_test(encodings, n_samples=params.num_samples)
            avg_esize = (esize_x+esize_y)/2
            avg_pval = (pval_x+pval_y)/2
            writer_exp3.writerow([f'{test_name}:{test_type}', esize_x, pval_x, esize_y, pval_y, avg_esize, avg_pval])
            
            log.info('\n\n')
            log.info('Computing general cosine similarities between cells (contextual)')
            vals = get_general_vals(encodings, n_samples=params.num_samples)
            writer_gen.writerow([f'{test_name}:{test_type}', vals['X_AXonAY'], vals['X_BXonBY'],
                                vals['Y_AXonAY'], vals['Y_BXonBY'], vals['X_AonB'],
                                vals['Y_AonB'], vals['X_ABXonABY'], vals['Y_ABXonABY']])

        f_exp1.flush()
        f_exp2.flush()
        f_exp3.flush()
        f_gen.flush()
