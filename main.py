#!/usr/bin/env python

from attrdict import AttrDict
from configargparse import ArgumentParser, YAMLConfigFileParser
import json
from os import makedirs, path
import torch
from scripts import(
    BiasTest, Writer, utils
)
from scripts.models import TYPE2WRAPPER

def load_eval_params():
    parser = ArgumentParser(config_file_parser_class=YAMLConfigFileParser)
    parser.add_argument('-c', '--config', is_config_file=True, required=True, type=str, help='config file path')
    parser.add_argument('--out_dir', type=str, default='results', help='path to save results')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--val_workers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--num_samples', type=int, default=100000, help='num/samples for p-val permutation test')
    parser.add_argument('--test2features_path', type=str, required=True, help='path to JSON file of features, stored by model and test')
    parser.add_argument('--tests', nargs='+', required=True, help='paths to tests to run')
    parser.add_argument('--model_type', type=str, required=True, choices=['lxmert', 'visualbert', 'vilbert', 'vlbert'])
    parser.add_argument('--model_archive', type=str, required=True, help='path to saved model to load')
    parser.add_argument('--max_seq_length', type=int, default=36)
    
    # add model-specific arguments
    model_type = parser.parse_known_args()[0].model_type
    TYPE2WRAPPER[model_type].add_model_args(parser)
    args = parser.parse_args()

    # additional arguments, check dirs
    args.num_gpus = torch.cuda.device_count()
    params = AttrDict({k:getattr(args, k) for k in vars(args)})

    # make model directories
    makedirs(params.out_dir, exist_ok=True)
    makedirs(path.join(params.out_dir, params.model_type), exist_ok=True)
    return params

def main():
    # load params and set up logging
    params = load_eval_params()
    log, save_dir, _ = utils.setup_logging_results(params)

    # load model wrapper
    model_wrapper = TYPE2WRAPPER[params.model_type](params)
    test2features = json.load(open(params.test2features_path))

    # load and run tests
    writer = Writer(save_dir)
    for bias_test_fp in params.tests:
        log.info(f'Loading {bias_test_fp}')
        with open(bias_test_fp, 'r') as f:
            test_data = json.load(f)

        test = BiasTest(params, test_data, test2features.get(params.model_type))
        #log.info(f'Total number of unique images: {test.get_num_unique_images()}')
        encodings = test.encode_data(model_wrapper)
        
        log.info('Running experiment 1: union across attribute A_{X} and A_{Y}')
        results = test.run_weat_union(encodings, params.num_samples)
        for test_type, (esize, pval) in results.items():
            writer.add_results_exp1(test.test_name, test_type, esize, pval)

        log.info('\n\n')
        log.info('Running experiment 2: corresponding attributes s(X, A_{X}, B_{X}) and s(Y, A_{Y}, B_{Y})')
        results = test.run_weat_specific(encodings, params.num_samples)
        for test_type, (esize, pval) in results.items():
            writer.add_results_exp2(test.test_name, test_type, esize, pval)

        log.info('\n\n')
        log.info('Running experiment 3: intra-target across target-specific attributes')
        results = test.run_weat_intra(encodings, params.num_samples)
        for test_type, (esize_x, pval_x, esize_y, pval_y) in results.items():
            writer.add_results_exp3(test.test_name, test_type, esize_x, pval_x, esize_y, pval_y)
        
        if 'sentence' in test.test_types or 'sent' in test.test_types: # TODO all sentence or all sent
            log.info('Running experiment 4a: masked language and 4b: masked vision')
            results = test.run_weat_mask(encodings, params.num_samples)
            mask_t_esize, mask_t_pval, test_type = results['mask_t']
            writer.add_results_exp4_mask_t(test.test_name, test_type, mask_t_esize, mask_t_pval)
            
            if 'mask_v' in results:
                mask_v_esize,mask_v_pval,test_type = results['mask_v']
                writer.add_results_exp4_mask_v(test.test_name, test_type, mask_v_esize, mask_v_pval)
        
        writer.flush()
    writer.close()

if __name__ == '__main__':
    main()