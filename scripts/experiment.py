import csv
import re
from os import path
from copy import deepcopy
import yaml
from attrdict import AttrDict
from typing import Dict, List
from torch import nn

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
    def __init__(self, test_data: Dict, test_filepath: str, params: AttrDict,
                 image_features_path: str):
        self.dataset = test_data['dataset']
        self.test_name = self.format_test_name(test_filepath)
        self.test_types = test_data['test_types']
        
        # copy for each separate dataloader
        params_targ_X = deepcopy(params)
        params_targ_Y = deepcopy(params)
        params_attr_AX = deepcopy(params)
        params_attr_AY = deepcopy(params)
        params_attr_BX = deepcopy(params)
        params_attr_BY = deepcopy(params)
    
        params_targ_X.update({'data_type' : 'target',
                              'category' : test_data['targ1']['category'],
                              'captions' : test_data['targ1']['captions'],
                              'images' : test_data['targ1']['images']})
        params_targ_Y.update({'data_type' : 'target',
                              'category' : test_data['targ2']['category'],
                              'captions' : test_data['targ2']['captions'],
                              'images' : test_data['targ2']['images']})
        
        
        category_X, category_Y = params_targ_X['category'], params_targ_Y['category']
        params_attr_AX.update({'data_type' : 'attr',
                                'category' : test_data['attr1']['category'],
                                'captions' : test_data['attr1']['captions'],
                                'images' : test_data['attr1'][f'{category_X}_Images']})
        params_attr_AY.update({'data_type' : 'attr',
                                'category' : test_data['attr1']['category'],
                                'captions' : test_data['attr1']['captions'],
                                'images' : test_data['attr1'][f'{category_Y}_Images']})
        params_attr_BX.update({'data_type' : 'attr',
                                'category' : test_data['attr2']['category'],
                                'captions' : test_data['attr2']['captions'],
                                'images' : test_data['attr2'][f'{category_X}_Images']})
        params_attr_BY.update({'data_type' : 'attr',
                                'category' : test_data['attr2']['category'],
                                'captions' : test_data['attr2']['captions'],
                                'images' : test_data['attr2'][f'{category_Y}_Images']})
        # model-specific datasets
        self.dataloader_targ_X = self.create_dataloader(params_targ_X, image_features_path)
        self.dataloader_targ_Y = self.create_dataloader(params_targ_Y, image_features_path)
        self.dataloader_attr_AX = self.create_dataloader(params_attr_AX, image_features_path)
        self.dataloader_attr_AY = self.create_dataloader(params_attr_AY, image_features_path)
        self.dataloader_attr_BX = self.create_dataloader(params_attr_BX, image_features_path)
        self.dataloader_attr_BY = self.create_dataloader(params_attr_BY, image_features_path)

        self.category_targ_X = test_data['targ1']['category']
        self.category_targ_Y = test_data['targ2']['category']
        self.category_attr_A = test_data['attr1']['category']
        self.category_attr_B = test_data['attr2']['category']

        self.context_indices = set()
        self.tokenize = self.dataloader_targ_X.dataset.tokenizer.tokenize
        self.convert_tokens_to_ids = self.dataloader_targ_X.dataset.tokenizer.convert_tokens_to_ids

        if 'word' in self.test_types:
            captions = list(test_data['targ1']['captions'].values()) +\
                       list(test_data['targ2']['captions'].values()) +\
                       list(test_data['attr1']['captions'].values()) +\
                       list(test_data['attr2']['captions'].values())
            for word in captions:
                subtokens = self.tokenize(word)
                self.context_indices.update(self.convert_tokens_to_ids([subtokens[-1]]))
            
        elif 'contextual_words' in test_data:
            for word in test_data['contextual_words']:
                subtokens = self.tokenize(word)
                self.context_indices.update(self.convert_tokens_to_ids([subtokens[-1]]))
        else:
            raise Exception('Context words missing!')
                
    def get_num_unique_images(self):
        return self.dataloader_targ_X.dataset.getNumUniqueImages() + \
                self.dataloader_targ_Y.dataset.getNumUniqueImages() + \
                self.dataloader_attr_AX.dataset.getNumUniqueImages() + \
                self.dataloader_attr_AY.dataset.getNumUniqueImages() + \
                self.dataloader_attr_BX.dataset.getNumUniqueImages() + \
                self.dataloader_attr_BY.dataset.getNumUniqueImages()
                    
    def format_test_name(self, test_filepath: str):
        test_name = re.sub('sent-|.jsonl', '', path.basename(test_filepath))
        test_name = re.sub('one_sentence|one_word', '', test_name)
        return test_name
        
    def create_dataloader(self, params: AttrDict, image_features_fp: str):
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

    def encode(self, encoder: nn.Module):
        # targets w/ corresponding images
        encoded_X, contextual_X = encoder.encode(self.dataloader_targ_X, self.context_indices)
        encoded_Y, contextual_Y = encoder.encode(self.dataloader_targ_Y, self.context_indices)
        
        # attribute A with images corresponding to targets X and Y
        encoded_attr_AX, contextual_AX = encoder.encode(self.dataloader_attr_AX, self.context_indices)
        encoded_attr_AY, contextual_AY = encoder.encode(self.dataloader_attr_AY, self.context_indices)
        
        # attribute B with images corresponding to targets X and Y
        encoded_attr_BX, contextual_BX = encoder.encode(self.dataloader_attr_BX, self.context_indices)
        encoded_attr_BY, contextual_BY = encoder.encode(self.dataloader_attr_BY, self.context_indices)
        
        encodings = {'targ_X' : encoded_X,
                     'targ_Y' : encoded_Y,
                     'attr_AX' : encoded_attr_AX,
                     'attr_AY' : encoded_attr_AY,
                     'attr_BX' : encoded_attr_BX,
                     'attr_BY' : encoded_attr_BY,
                     'contextual_targ_X' : contextual_X,
                     'contextual_targ_Y' : contextual_Y,
                     'contextual_attr_AX' : contextual_AX,
                     'contextual_attr_AY' : contextual_AY,
                     'contextual_attr_BX' : contextual_BX,
                     'contextual_attr_BY' : contextual_BY
        }
        return encodings

    def _get_revelant_encodings(self, test_type: str, encodings: Dict):
        if test_type == 'word' or test_type == 'sentence':
            X =  encodings['targ_X']
            Y =  encodings['targ_Y']
            AX = encodings['attr_AX']
            AY = encodings['attr_AY']
            BX = encodings['attr_BX']
            BY = encodings['attr_BY']
        elif test_type == 'contextual':
            X =  encodings['contextual_targ_X']
            Y =  encodings['contextual_targ_Y']
            AX = encodings['contextual_attr_AX']
            AY = encodings['contextual_attr_AY']
            BX = encodings['contextual_attr_BX']
            BY = encodings['contextual_attr_BY']
        else:
            raise Exception(f'Unknown test type: {test_type}')
        return X, Y, AX, AY, BX, BY
    
    def run_weat_union(self, encodings: Dict, num_samples: int, cat_X: str, cat_Y: str,
                       cat_A: str, cat_B: str):
        results = {}
        for test_type in self.test_types:
            X, Y, AX, AY, BX, BY = self._get_revelant_encodings(test_type, encodings)
            esize, pval = weat_union.run_test(X, Y, AX, AY, BX, BY, num_samples,
                                              cat_X, cat_Y, cat_A, cat_B)
            results[test_type] = (esize, pval)
        return results

    def run_weat_specific(self, encodings: Dict, num_samples: int, cat_X: str, cat_Y: str,
                       cat_A: str, cat_B: str):
        results = {}
        for test_type in self.test_types:
            X, Y, AX, AY, BX, BY = self._get_revelant_encodings(test_type, encodings)
            esize, pval = weat_specific.run_test(X, Y, AX, AY, BX, BY, num_samples,
                                                 cat_X, cat_Y, cat_A, cat_B)
            results[test_type] = (esize, pval)
        return results
        
    def run_weat_intra(self, encodings: Dict, num_samples: int, cat_X: str, cat_Y: str,
                       cat_A: str, cat_B: str):
        results = {}
        for test_type in self.test_types:
            X, Y, AX, AY, BX, BY = self._get_revelant_encodings(test_type, encodings)
            esize_x, pval_x, esize_y, pval_y =\
                        weat_intra.run_test(X, Y, AX, AY, BX, BY, num_samples,
                                            cat_X, cat_Y, cat_A, cat_B)
            results[test_type] = (esize_x, pval_x, esize_y, pval_y)
        return results

    def get_general_vals(self, encodings: Dict, num_samples: int): # TODO rename
        results = {}
        for test_type in self.test_types:
            X, Y, AX, AY, BX, BY = self._get_revelant_encodings(test_type, encodings)
            vals = get_general_vals(X, Y, AX, AY, BX, BY, n_samples=num_samples)
            results[test_type] = vals
        return results
