from copy import deepcopy
from attrdict import AttrDict
from typing import Dict
import torch
from torch import nn
from warnings import warn

from .weat.weat_images_union import run_test as weat_union
from .weat.weat_images_targ_specific import run_test as weat_specific
from .weat.weat_images_intra_targ import run_test as weat_intra
from .weat.general_vals import get_general_vals

from .dataloaders.dataset_wrappers import create_dataset
from .dataloaders.bias_dataloader import BiasDataLoader

class BiasTest:
    def __init__(
        self,
        params: AttrDict,
        test_data: Dict,
        preextracted_features: Dict[str, str]=None
        ):
        self.test_name = test_data['test_name']
        self.dataset_name = test_data['dataset']
        self.test_types = test_data['test_types']
        print(f'Loading test {self.test_name}')
        
        if preextracted_features:
            image_features_path_or_dir = preextracted_features[self.dataset_name][self.test_name]
        else:
            image_features_path_or_dir = None
        self.dataset = create_dataset(
            params=deepcopy(params),
            captions=test_data['targ1']['captions'],
            images=test_data['targ1']['images'],
            image_features_path_or_dir=image_features_path_or_dir
        )

        # add uncased versions of all contextual words as well
        test_data['contextual_words'].extend([w.lower() for w in test_data['contextual_words']])
        
        # copy for each separate dataloader
        self.category_X = test_data['targ1']['category']
        self.category_Y = test_data['targ2']['category']
        self.category_A = test_data['attr1']['category']
        self.category_B = test_data['attr2']['category']

        self.dataloader_targ_X = BiasDataLoader(
            dataset=self.dataset,
            batch_size=params.batch_size,
            num_gpus=params.num_gpus,
            category=self.category_X,
            contextual_words=test_data['contextual_words'],
            )
        self.dataloader_targ_Y = BiasDataLoader(
            dataset=self.dataset,
            batch_size=params.batch_size,
            num_gpus=params.num_gpus,
            category=self.category_Y,
            contextual_words=test_data['contextual_words'],
            )
        self.dataloader_attr_AX = BiasDataLoader(
            dataset=self.dataset,
            batch_size=params.batch_size,
            num_gpus=params.num_gpus,
            category=self.category_A,
            contextual_words=test_data['contextual_words'],
            )
        self.dataloader_attr_AY = BiasDataLoader(
            dataset=self.dataset,
            batch_size=params.batch_size,
            num_gpus=params.num_gpus,
            category=self.category_A,
            contextual_words=test_data['contextual_words'],
            )
        self.dataloader_attr_BX = BiasDataLoader(
            dataset=self.dataset,
            batch_size=params.batch_size,
            num_gpus=params.num_gpus,
            category=self.category_B,
            contextual_words=test_data['contextual_words'],
            )
        self.dataloader_attr_BY = BiasDataLoader(
            dataset=self.dataset,
            batch_size=params.batch_size,
            num_gpus=params.num_gpus,
            category=self.category_B,
            contextual_words=test_data['contextual_words'],
            )
        self.dataloaders = [
            self.dataloader_targ_X, self.dataloader_targ_Y,
            self.dataloader_attr_AX, self.dataloader_attr_AY,
            self.dataloader_attr_BX, self.dataloader_attr_BY
            ]
        
    def dataloaders(self):
        for dataloader in self.dataloaders:
            yield dataloader
        
    @torch.no_grad()
    def encode_data(self, model: nn.Module):
        encodings = {}
        
        # targets w/ corresponding images
        encoded_X, encoded_X_mask_t, encoded_X_mask_v = model.encode(self.dataloader_targ_X)
        encoded_Y, encoded_Y_mask_t, encoded_Y_mask_v = model.encode(self.dataloader_targ_Y)
        
        # targets w/ corresponding images
        encoded_X, encoded_X_mask_t, encoded_X_mask_v = model.encode(self.dataloader_targ_X)
        encoded_Y, encoded_Y_mask_t, encoded_Y_mask_v = model.encode(self.dataloader_targ_Y)
        
        # attribute A with images corresponding to targets X and Y
        encoded_AX, encoded_AX_mask_t, encoded_AX_mask_v = model.encode(self.dataloader_attr_AX)
        encoded_AY, encoded_AY_mask_t, encoded_AY_mask_v = model.encode(self.dataloader_attr_AY)
        
        # attribute B with images corresponding to targets X and Y
        encoded_BX, encoded_BX_mask_t, encoded_BX_mask_v = model.encode(self.dataloader_attr_BX)
        encoded_BY, encoded_BY_mask_t, encoded_BY_mask_v = model.encode(self.dataloader_attr_BY)
            
        encodings = {
            'targ_X' : encoded_X['full_seq'],
            'targ_Y' : encoded_Y['full_seq'],
            'attr_AX' : encoded_AX['full_seq'],
            'attr_AY' : encoded_AY['full_seq'],
            'attr_BX' : encoded_BX['full_seq'],
            'attr_BY' : encoded_BY['full_seq'],
            'targ_X_mask_t' : encoded_X_mask_t['full_seq'],
            'targ_Y_mask_t' : encoded_Y_mask_t['full_seq'],
            'attr_AX_mask_t' : encoded_AX_mask_t['full_seq'],
            'attr_AY_mask_t' : encoded_AY_mask_t['full_seq'],
            'attr_BX_mask_t' : encoded_BX_mask_t['full_seq'],
            'attr_BY_mask_t' : encoded_BY_mask_t['full_seq'],
            'targ_X_mask_v' : encoded_X_mask_v['full_seq'],
            'targ_Y_mask_v' : encoded_Y_mask_v['full_seq'],
            'attr_AX_mask_v' : encoded_AX_mask_v['full_seq'],
            'attr_AY_mask_v' : encoded_AY_mask_v['full_seq'],
            'attr_BX_mask_v' : encoded_BX_mask_v['full_seq'],
            'attr_BY_mask_v' : encoded_BY_mask_v['full_seq'],
            'contextual_targ_X' : encoded_X['contextual'],
            'contextual_targ_Y' : encoded_Y['contextual'],
            'contextual_attr_AX' : encoded_AX['contextual'],
            'contextual_attr_AY' : encoded_AY['contextual'],
            'contextual_attr_BX' : encoded_BX['contextual'],
            'contextual_attr_BY' : encoded_BY['contextual']
        }
        return encodings

    @torch.no_grad()
    def predict_words(self, model: nn.Module):
        print("attr AX")
        model.predict_words(self.dataloader_attr_AX)
        print("\n\nattr AY")
        model.predict_words(self.dataloader_attr_AY)
        print("attr BX")
        model.predict_words(self.dataloader_attr_BX)
        print("\n\nattr BY")
        model.predict_words(self.dataloader_attr_BY)
        exit()
        
    
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
        elif test_type == 'mask_t':
            X =  encodings['targ_X_mask_t']
            Y =  encodings['targ_Y_mask_t']
            AX = encodings['attr_AX_mask_t']
            AY = encodings['attr_AY_mask_t']
            BX = encodings['attr_BX_mask_t']
            BY = encodings['attr_BY_mask_t']
        elif test_type == 'mask_v':
            X =  encodings['targ_X_mask_v']
            Y =  encodings['targ_Y_mask_v']
            AX = encodings['attr_AX_mask_v']
            AY = encodings['attr_AY_mask_v']
            BX = encodings['attr_BX_mask_v']
            BY = encodings['attr_BY_mask_v']
        else:
            raise Exception(f'Unknown test type: {test_type}')
        return X, Y, AX, AY, BX, BY
    
    
    def run_weat_union(self, encodings: Dict, num_samples: int):
        results = {}
        for test_type in self.test_types:
            X, Y, AX, AY, BX, BY = self._get_revelant_encodings(test_type, encodings)
            esize, pval = weat_union(X, Y, AX, AY, BX, BY, num_samples,
                                     self.category_X, self.category_Y,
                                     self.category_A, self.category_B)
            results[test_type] = (esize, pval)
        return results

    def run_weat_specific(self, encodings: Dict, num_samples: int):
        results = {}
        for test_type in self.test_types:
            X, Y, AX, AY, BX, BY = self._get_revelant_encodings(test_type, encodings)
            esize, pval = weat_specific(X, Y, AX, AY, BX, BY, num_samples,
                                        self.category_X, self.category_Y,
                                        self.category_A, self.category_B)
            results[test_type] = (esize, pval)
        return results
        
    def run_weat_intra(self, encodings: Dict, num_samples: int):
        results = {}
        for test_type in self.test_types:
            X, Y, AX, AY, BX, BY = self._get_revelant_encodings(test_type, encodings)
            esize_x, pval_x, esize_y, pval_y =\
                        weat_intra(X, Y, AX, AY, BX, BY, num_samples,
                                   self.category_X, self.category_Y,
                                   self.category_A, self.category_B)
            results[test_type] = (esize_x, pval_x, esize_y, pval_y)
        return results

    def run_weat_mask(self, encodings: Dict, num_samples: int):
        results = {}
        for mask_type in ['mask_t', 'mask_v']:
            X, Y, AX, AY, BX, BY = self._get_revelant_encodings(mask_type, encodings)
            if len(X) == 0:
                warn(f'Length is zero for X for {mask_type}')
                continue
            test_type = 'word' if 'word' in self.test_types else 'sent'
            esize, pval = weat_union(X, Y, AX, AY, BX, BY, num_samples,
                                     self.category_X, self.category_Y,
                                     self.category_A, self.category_B)
            results[mask_type] = (esize, pval, test_type)
        return results
    
    def get_general_vals(self, encodings: Dict, num_samples: int): # TODO rename
        results = {}
        for test_type in self.test_types:
            X, Y, AX, AY, BX, BY = self._get_revelant_encodings(test_type, encodings)
            vals = get_general_vals(X, Y, AX, AY, BX, BY, n_samples=num_samples)
            results[test_type] = vals
        return results
