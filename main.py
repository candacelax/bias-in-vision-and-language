#!/usr/bin/env python

from os import path
import json
import re
import csv
from scripts import utils
from scripts.experiment import BiasTest

if __name__ == '__main__':
    # Load params and set up logging
    params = utils.load_params()
    log, timestamp = utils.setup_log(params)
    save_dir = path.join(params.out_dir, timestamp)
    encoder = utils.load_model(params)
    test2features = json.load(open(params.test2features))

    # Load and run tests
    with open(f'{save_dir}/exp1.csv', 'w') as f_exp1,\
         open(f'{save_dir}/exp2.csv', 'w') as f_exp2,\
         open(f'{save_dir}/exp3.csv', 'w') as f_exp3,\
         open(f'{save_dir}/general.csv', 'w') as f_gen:
        writer_exp1 = csv.writer(f_exp1, delimiter=',')
        writer_exp2 = csv.writer(f_exp2, delimiter=',')
        writer_exp3 = csv.writer(f_exp3, delimiter=',')
        writer_gen = csv.writer(f_gen, delimiter=',')

        # write headers
        writer_exp1.writerow(['test_name', 'esize', 'pval'])
        writer_exp2.writerow(['test_name', 'esize', 'pval'])
        writer_exp3.writerow(['test_name', 'esize_x', 'pval_x', 'esize_y', 'pval_y'])
        writer_gen.writerow(['test_name', 'X_AXonAY', 'X_BXonBY',
                             'Y_AXonAY', 'Y_BXonBY', 'X_AonB', 'Y_AonB',
                             'X_ABXonABY', 'Y_ABXonABY'])
        
        for bias_test_fp in params.tests:
            log.info(f'Loading {bias_test_fp}')
            with open(bias_test_fp, 'r') as f:
                test_data = json.load(f)
                dataset_name = test_data['dataset']
                test_basename = path.basename(bias_test_fp)

            image_features_fp = test2features[params.model_type][dataset_name][test_basename]
            test = BiasTest(test_data, bias_test_fp, params, image_features_fp)
            test_name = test.test_name
            
            log.info(f'Total number of unique images: {test.get_num_unique_images()}')
            encodings = test.encode(encoder)
            
            log.info('Running experiment 1: union across attribute A_{X} and A_{Y}')
            results = test.run_weat_union(encodings, params.num_samples,
                                          test.category_targ_X, test.category_targ_Y,
                                          test.category_attr_A, test.category_attr_B)
            for test_type, (esize, pval) in results.items():
                writer_exp1.writerow([f'{test_name}:{test_type}', esize, pval])
                
            log.info('\n\n')
            log.info('Running experiment 2: corresponding attributes s(X, A_{X}, B_{X}) and s(Y, A_{Y}, B_{Y})')
            results = test.run_weat_specific(encodings, params.num_samples,
                                             test.category_targ_X, test.category_targ_Y,
                                             test.category_attr_A, test.category_attr_B)
            for test_type, (esize, pval) in results.items():
                writer_exp2.writerow([f'{test_name}:{test_type}', esize, pval])
            
            log.info('\n\n')
            log.info('Running experiment 3: intra-target across target-specific attributes')
            results = test.run_weat_intra(encodings, params.num_samples,
                                          test.category_targ_X, test.category_targ_Y,
                                          test.category_attr_A, test.category_attr_B)
            for test_type, (esize_x, pval_x, esize_y, pval_y) in results.items():
                writer_exp3.writerow([f'{test_name}:{test_type}', esize_x, pval_x, esize_y, pval_y])
            print('done')
            
            log.info('\n\n')
            log.info('Computing general cosine similarities between cells')
            results = test.get_general_vals(encodings, params.num_samples)
            for test_type, vals in results.items():
                writer_gen.writerow([f'{test_name}:{test_type}', vals['X_AXonAY'], vals['X_BXonBY'],
                                     vals['Y_AXonAY'], vals['Y_BXonBY'], vals['X_AonB'],
                                     vals['Y_AonB'], vals['X_ABXonABY'], vals['Y_ABXonABY']]) 
        f_exp1.flush()
        f_exp2.flush()
        f_exp3.flush()
        f_gen.flush()
