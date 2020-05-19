import csv
import re
from os import path
from copy import deepcopy
import yaml

# VisualBERT
from visualbert.dataloaders.vcr import VCRLoader
from visualbert.dataloaders.bias_dataset import BiasDataset as BiasDatasetVisualBERT
# ViLBERT
from vilbert.datasets.bias_dataset import BiasLoader as BiasLoaderViLBERT
from vilbert.task_utils import LoadBiasDataset as LoadBiasDatasetViLBERT

from scripts.weat import weat_images_union as weat_union
from scripts.weat import weat_images_targ_specific as weat_specific
from scripts.weat import weat_images_intra_targ as weat_intra
from scripts.weat.general_vals import get_general_vals # TODO rename

class BiasTest:
    def __init__(self, test_data, test_filepath, params, image_features_path):
        self.dataset = test_data['dataset']
        self.test_name = self.format_test_name(test_filepath)
        self.test_types = test_data['test_types']

        # copy for each separate dataloader
        params_targ_X = deepcopy(params)
        params_targ_Y = deepcopy(params)
        params_attr_A_X = deepcopy(params)
        params_attr_A_Y = deepcopy(params)
        params_attr_B_X = deepcopy(params)
        params_attr_B_Y = deepcopy(params)
    
        params_targ_X.update({'data_type' : 'target',
                              'category' : test_data['targ1']['category'],
                              'captions' : test_data['targ1']['captions'],
                              'images' : test_data['targ1']['images']})
        params_targ_Y.update({'data_type' : 'target',
                              'category' : test_data['targ2']['category'],
                              'captions' : test_data['targ2']['captions'],
                              'images' : test_data['targ2']['images']})

        category_X, category_Y = params_targ_X['category'], params_targ_Y['category']
        params_attr_A_X.update({'data_type' : 'attr',
                                'category' : test_data['attr1']['category'],
                                'captions' : test_data['attr1']['captions'],
                                'images' : test_data['attr1'][f'{category_X}_Images']})
        params_attr_A_Y.update({'data_type' : 'attr',
                                'category' : test_data['attr1']['category'],
                                'captions' : test_data['attr1']['captions'],
                                'images' : test_data['attr1'][f'{category_Y}_Images']})
        params_attr_B_X.update({'data_type' : 'attr',
                                'category' : test_data['attr2']['category'],
                                'captions' : test_data['attr2']['captions'],
                                'images' : test_data['attr2'][f'{category_X}_Images']})
        params_attr_B_Y.update({'data_type' : 'attr',
                                'category' : test_data['attr2']['category'],
                                'captions' : test_data['attr2']['captions'],
                                'images' : test_data['attr2'][f'{category_Y}_Images']})

        # model-specific datasets
        self.dataloader_targ_X = self.create_dataloader(params_targ_X, image_features_path)
        self.dataloader_targ_Y = self.create_dataloader(params_targ_Y, image_features_path)
        self.dataloader_attr_A_X = self.create_dataloader(params_attr_A_X, image_features_path)
        self.dataloader_attr_A_Y = self.create_dataloader(params_attr_A_Y, image_features_path)
        self.dataloader_attr_B_X = self.create_dataloader(params_attr_B_X, image_features_path)
        self.dataloader_attr_B_Y = self.create_dataloader(params_attr_B_Y, image_features_path)

        self.category_targ_X = test_data['targ1']['category']
        self.category_targ_Y = test_data['targ2']['category']
        self.category_attr_A = test_data['attr1']['category']
        self.category_attr_B = test_data['attr2']['category']
        
    def format_test_name(self, test_filepath):
        test_name = re.sub('sent-|.jsonl', '', path.basename(test_filepath))
        test_name = re.sub('one_sentence|one_word', '', test_name)
        return test_name
        
    def create_dataloader(self, params, image_features_fp):
        if params.model_type == 'visualbert':
            params.chunk_path = image_features_fp
            dataset = BiasDatasetVisualBERT(params)
            loader_params = {'batch_size': params.batch_size // params.num_gpus,
                             'num_gpus': params.num_gpus,
                             'num_workers': params.num_workers}
            return VCRLoader.from_dataset(dataset, **loader_params)
        else:
            with open(params.task_cfg, 'r') as f:
                task_cfg = yaml.load(f, Loader=yaml.FullLoader)
            params.features_fpath = image_features_fp
            return LoadBiasDatasetViLBERT(params, task_cfg)

    def encode(self, encoder):
        # targets w/ corresponding images
        encoded_X, contextual_X = encoder.encode(self.dataloader_targ_X)
        encoded_Y, contextual_Y = encoder.encode(self.dataloader_targ_Y)
        
        # attribute A with images corresponding to targets X and Y
        encoded_attr_A_X, contextual_A_X = encoder.encode(self.dataloader_attr_A_X)
        encoded_attr_A_Y, contextual_A_Y = encoder.encode(self.dataloader_attr_A_Y)
        
        # attribute B with images corresponding to targets X and Y
        encoded_attr_B_X, contextual_B_X = encoder.encode(self.dataloader_attr_B_X)
        encoded_attr_B_Y, contextual_B_Y = encoder.encode(self.dataloader_attr_B_Y)
        
        encodings = {'targ_X' : encoded_X,
                     'targ_Y' : encoded_Y,
                     'attr_A_X' : encoded_attr_A_X,
                     'attr_A_Y' : encoded_attr_A_Y,
                     'attr_B_X' : encoded_attr_B_X,
                     'attr_B_Y' : encoded_attr_B_Y,
                     'contextual_targ_X' : contextual_X,
                     'contextual_targ_Y' : contextual_Y,
                     'contextual_attr_A_X' : contextual_A_X,
                     'contextual_attr_A_Y' : contextual_A_Y,
                     'contextual_attr_B_X' : contextual_B_X,
                     'contextual_attr_B_Y' : contextual_B_Y
        }
        return encodings

    def _get_revelant_encodings(self, test_type, encodings):
        if test_type == 'word' or test_type == 'sentence':
            X =  encodings['targ_X']
            Y =  encodings['targ_Y']
            A_X = encodings['attr_A_X']
            A_Y = encodings['attr_A_Y']
            B_X = encodings['attr_B_X']
            B_Y = encodings['attr_B_Y']
        elif test_type == 'contextual':
            X =  encodings['contextual_targ_X']
            Y =  encodings['contextual_targ_Y']
            A_X = encodings['contextual_attr_A_X']
            A_Y = encodings['contextual_attr_A_Y']
            B_X = encodings['contextual_attr_B_X']
            B_Y = encodings['contextual_attr_B_Y']
        else:
            raise Exception(f'Unknown test type: {test_type}')
        return X, Y, A_X, A_Y, B_X, B_Y
    
    def run_weat_union(self, encodings, num_samples, cat_X, cat_Y, cat_A, cat_B):
        results = {}
        for test_type in self.test_types:
            X, Y, A_X, A_Y, B_X, B_Y = self._get_revelant_encodings(test_type, encodings)
            esize, pval = weat_union.run_test(X, Y, A_X, A_Y, B_X, B_Y, num_samples,
                                              cat_X, cat_Y, cat_A, cat_B)
            results[test_type] = (esize, pval)
        return results

    def run_weat_specific(self, encodings, num_samples, cat_X, cat_Y, cat_A, cat_B):
        results = {}
        for test_type in self.test_types:
            X, Y, A_X, A_Y, B_X, B_Y = self._get_revelant_encodings(test_type, encodings)
            esize, pval = weat_specific.run_test(X, Y, A_X, A_Y, B_X, B_Y, num_samples,
                                                 cat_X, cat_Y, cat_A, cat_B)
            results[test_type] = (esize, pval)
        return results
        
    def run_weat_intra(self, encodings, num_samples, cat_X, cat_Y, cat_A, cat_B):
        results = {}
        for test_type in self.test_types:
            X, Y, A_X, A_Y, B_X, B_Y = self._get_revelant_encodings(test_type, encodings)
            esize_x, pval_x, esize_y, pval_y =\
                        weat_intra.run_test(X, Y, A_X, A_Y, B_X, B_Y, num_samples,
                                            cat_X, cat_Y, cat_A, cat_B)
            results[test_type] = (esize_x, pval_x, esize_y, pval_y)
        return results

    def get_general_vals(self, encodings, num_samples): # TODO rename
        results = {}
        for test_type in self.test_types:
            X, Y, A_X, A_Y, B_X, B_Y = self._get_revelant_encodings(test_type, encodings)
            vals = get_general_vals(X, Y, A_X, A_Y, B_X, B_Y, n_samples=params.num_samples)
            results[test_type] = vals
        return results
