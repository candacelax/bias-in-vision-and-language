import csv
import re
from os import path
from copy import deepcopy
from attrdict import AttrDict
from typing import Dict, List
import torch
from torch import nn
from torch.utils.data import DataLoader

import scripts
import datasets

class BiasTest:
    def __init__(self, test_data: Dict, test_filepath: str, params: AttrDict,
                 image_features_path: str, obj_list: List):
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
        self.dataloader_targ_X = \
                    self.create_dataloader(params_targ_X, image_features_path, obj_list)
        self.dataloader_targ_Y = \
                    self.create_dataloader(params_targ_Y, image_features_path, obj_list)
        self.dataloader_attr_AX = \
                    self.create_dataloader(params_attr_AX, image_features_path, obj_list)
        self.dataloader_attr_AY = \
                    self.create_dataloader(params_attr_AY, image_features_path, obj_list)
        self.dataloader_attr_BX = \
                    self.create_dataloader(params_attr_BX, image_features_path, obj_list)
        self.dataloader_attr_BY = \
                    self.create_dataloader(params_attr_BY, image_features_path, obj_list)



        self.category_X = self.dataloader_targ_X.category
        self.category_Y = self.dataloader_targ_Y.category
        self.category_A = self.dataloader_attr_AX.category
        self.category_B = self.dataloader_attr_BX.category
        
        
        self.contextual_word_ids = set()
        if 'word' in self.test_types:
            captions = list(test_data['targ1']['captions'].values()) +\
                       list(test_data['targ2']['captions'].values()) +\
                       list(test_data['attr1']['captions'].values()) +\
                       list(test_data['attr2']['captions'].values())
            for word in captions:
                subtokens = self.tokenize(word)
                self.contextual_word_ids.update(self.convert_tokens_to_ids([subtokens[-1]]))
            
        elif 'contextual_words' in test_data:
            for word in test_data['contextual_words']:
                subtokens = self.tokenize(word)
                self.contextual_word_ids.update(self.convert_tokens_to_ids([subtokens[-1]]))
        else:
            raise Exception('Context words missing!')

    def dataloaders(self):
        for dataloader in [self.dataloader_targ_X, self.dataloader_targ_Y,
                           self.dataloader_attr_AX, self.dataloader_attr_AY,
                           self.dataloader_attr_BX, self.dataloader_attr_BY]:
            yield dataloader        
        
    # def get_num_unique_images(self):
    #     return self.dataloader_targ_X.dataset.getNumUniqueImages() + \
    #             self.dataloader_targ_Y.dataset.getNumUniqueImages() + \
    #             self.dataloader_attr_AX.dataset.getNumUniqueImages() + \
    #             self.dataloader_attr_AY.dataset.getNumUniqueImages() + \
    #             self.dataloader_attr_BX.dataset.getNumUniqueImages() + \
    #             self.dataloader_attr_BY.dataset.getNumUniqueImages()
                    
    def format_test_name(self, test_filepath: str):
        test_name = re.sub('sent-|.jsonl', '', path.basename(test_filepath))
        test_name = re.sub('one_sentence|one_word', '', test_name)
        return test_name
        
    def create_dataloader(self, params: AttrDict, image_features_fp: str, obj_list: List):
        if params.model_type == 'visualbert':
            params.chunk_path = image_features_fp
            dataset = datasets.BiasDatasetVisualBERT(params)
            loader_params = {'batch_size': params.batch_size // params.num_gpus,
                             'num_gpus': params.num_gpus,
                             'num_workers': params.num_workers}
            dataloader = datasets.VCRLoader.from_dataset(dataset, **loader_params)
            if not hasattr(self, 'tokenize'):
                self.tokenize = dataloader.dataset.tokenizer.tokenize
                self.convert_tokens_to_ids = dataloader.dataset.tokenizer.convert_tokens_to_ids
                self.convert_ids_to_tokens = dataloader.dataset.tokenizer.convert_ids_to_tokens
            dataloader.category = dataloader.dataset.category
        else:
            dataloader = datasets.BiasLoaderViLBERT(bert_model_name=params.bert_model,
                                                    captions=params.captions,
                                                    images=params.images,
                                                    dataset_type=self.dataset,
                                                    category=params.category,
                                                    image_features_fp=image_features_fp,
                                                    obj_list=obj_list,
                                                    seq_len=params.max_seq_length,
                                                    batch_size=params.batch_size,
                                                    num_workers=params.num_workers,
                                                    cuda=True)
            if not hasattr(self, 'tokenize'):
                self.tokenize = dataloader.tokenizer.tokenize
                self.convert_tokens_to_ids = dataloader.tokenizer.convert_tokens_to_ids
                self.convert_ids_to_tokens = dataloader.tokenizer.convert_ids_to_tokens
        return dataloader
                  
    @torch.no_grad()
    def encode_test_data(self, model: nn.Module):
        model.eval()
        f_encode = self._encode_visualbert if isinstance(model, scripts.VisualBERTModelWrapper) \
                   else self._encode_vilbert
        
        # targets w/ corresponding images
        encoded_X, encoded_X_mask_t, encoded_X_mask_v = f_encode(model, self.dataloader_targ_X)
        encoded_Y, encoded_Y_mask_t, encoded_Y_mask_v = f_encode(model, self.dataloader_targ_Y)
        
        # attribute A with images corresponding to targets X and Y
        encoded_AX, encoded_AX_mask_t, encoded_AX_mask_v = f_encode(model, self.dataloader_attr_AX)
        encoded_AY, encoded_AY_mask_t, encoded_AY_mask_v = f_encode(model, self.dataloader_attr_AY)
        
        # attribute B with images corresponding to targets X and Y
        encoded_BX, encoded_BX_mask_t, encoded_BX_mask_v = f_encode(model, self.dataloader_attr_BX)
        encoded_BY, encoded_BY_mask_t, encoded_BY_mask_v = f_encode(model, self.dataloader_attr_BY)
            
        encodings = {'targ_X' : encoded_X['full_seq'],
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

    def _encode_visualbert(self, model: nn.Module, dataloader: DataLoader):
        enc_full_seq = {} # either word or sentence (depending on input)
        enc_contextual = {} # word in context
        enc_mask_t_full_seq = {} # relevant text indices masked
        enc_mask_t_contextual = {}
        enc_mask_v_full_seq = {} # relevant image regions masked
        enc_mask_v_contextual = {}
        mask_id = self.convert_tokens_to_ids(['[MASK]'])[0]

        for batch in dataloader:
            # 1. with full access to all tokens and all image regions
            output = model.step(batch, eval_mode=True, output_all_encoded_layers=True)
            sequence_output = output['sequence_output'][-1]
            input_ids = batch['bert_input_ids'].detach().cpu()

            # 2. with full access to all regions and masked language tokens
            masked_input_ids = batch['bert_input_ids'].clone().detach()
            for c in self.contextual_word_ids: # mask relevant contextual words
                masked_input_ids[masked_input_ids == c] = mask_id
            batch['bert_input_ids'] = masked_input_ids
            mask_t_sequence_output = output['sequence_output'][-1]

            for idx in range(len(sequence_output)):
                seq_out = sequence_output[idx]
                mask_t_seq_out = mask_t_sequence_output[idx]
                
                index_of_target_id = self._get_contextual_target_id(input_ids[idx])
                # input_ids = .cpu().tolist()
                # target_ids = list(self.contextual_word_ids.intersection(input_ids))
                # #x = ' '.join(dataloader.dataset.tokenizer.convert_ids_to_tokens(input_ids))
                # target_id = sorted(target_ids)[-1] # take final subword idx
                # index_of_target_id = input_ids.index(target_id)
                    
                # take 0-th dim corresponding to CLS token (for words + sents)
                enc_full_seq[len(enc_full_seq)] = seq_out[0,:].detach().cpu()
                enc_contextual[len(enc_contextual)] = seq_out[index_of_target_id,:].detach().cpu()
                enc_mask_t_full_seq[len(enc_mask_t_full_seq)] = mask_t_seq_out[0,:].detach().cpu()

        enc = {'full_seq' : enc_full_seq,
               'contextual' : enc_contextual}
        enc_mask_v = {'full_seq' : enc_mask_v_full_seq,
                      'contextual' : enc_mask_v_contextual}
        enc_mask_t = {'full_seq' : enc_mask_t_full_seq,
                      'contextual' : enc_mask_t_contextual}
        return enc, enc_mask_t, enc_mask_v

    def _encode_vilbert(self, model: nn.Module, dataloader: DataLoader):
        enc_full_seq = {} # either word or sentence (depending on input)
        enc_contextual = {} # word in context
        enc_mask_t_full_seq = {} # relevant text indices masked
        enc_mask_t_contextual = {}
        enc_mask_v_full_seq = {} # relevant image regions masked
        enc_mask_v_contextual = {}
        mask_id = self.convert_tokens_to_ids(['[MASK]'])[0]
        
        for batch in dataloader:
            batch = tuple(t.cuda(non_blocking=True) for t in batch)

            input_ids, input_mask, segment_ids, lm_label_ids, image_feat, image_loc, \
                image_label, image_mask, image_ids, coattention_mask,\
                masked_image_feat, masked_image_label = batch

            # 1. with full access to all tokens and all image regions
            output = model(input_ids, image_feat, image_loc, segment_ids,
                           input_mask, image_mask, coattention_mask,
                           return_sequence_output=True)
            sequence_output_t, sequence_output_v = output[-2:]

            # 2. with full access to all tokens and masked image regions
            masked_v_output = model(input_ids, masked_image_feat, image_loc, segment_ids,
                                    input_mask, image_mask, coattention_mask,
                                    return_sequence_output=True)
            masked_v_sequence_output_t, masked_v_sequence_output_v = output[-2:]

            # 3. with full access to all regions and masked language tokens
            # TODO probably move to dataloader
            masked_t_input_ids = input_ids.clone().detach()
            for c in self.contextual_word_ids: # mask relevant contextual words
                masked_t_input_ids[masked_t_input_ids == c] = mask_id

            masked_t_output = model(masked_t_input_ids, image_feat, image_loc, segment_ids,
                                    input_mask, image_mask, coattention_mask,
                                    return_sequence_output=True)
            masked_t_sequence_output_t, masked_t_sequence_output_v = output[-2:]
                

            for idx in range(len(sequence_output_t)):
                # full access to vision & lang
                seq_out_t = sequence_output_t[idx]
                seq_out_v = sequence_output_v[idx]

                # masked image regions
                masked_v_seq_out_t = masked_v_sequence_output_t[idx]
                masked_v_seq_out_v = masked_v_sequence_output_v[idx]

                # masked text ids
                masked_t_seq_out_t = masked_t_sequence_output_t[idx]
                masked_t_seq_out_v = masked_t_sequence_output_v[idx]

                index_of_target_id = self._get_contextual_target_id(input_ids[idx])
                    
                # take 0-th dim corresponding to CLS token (for words + sents)
                enc_idx = len(enc_full_seq)
                cat_seq_out = torch.cat((seq_out_t[0,:], seq_out_v[0,:]), dim=0).detach().cpu()
                cat_masked_v_seq_out = torch.cat((masked_v_seq_out_t[0,:],
                                                  masked_v_seq_out_v[0,:]), dim=0).detach().cpu()
                cat_masked_t_seq_out = torch.cat((masked_t_seq_out_t[0,:],
                                                  masked_t_seq_out_v[0,:]), dim=0).detach().cpu()

                # TODO decide about contextual for image
                enc_full_seq[enc_idx] = cat_seq_out
                enc_contextual[enc_idx] = seq_out_t[index_of_target_id,:].detach().cpu()
                enc_mask_v_full_seq[enc_idx] = cat_masked_v_seq_out
                enc_mask_t_full_seq[enc_idx] = cat_masked_t_seq_out

        enc = {'full_seq' : enc_full_seq,
               'contextual' : enc_contextual}
        enc_mask_v = {'full_seq' : enc_mask_v_full_seq,
                      'contextual' : enc_mask_v_contextual}
        enc_mask_t = {'full_seq' : enc_mask_t_full_seq,
                      'contextual' : enc_mask_t_contextual}
        return enc, enc_mask_t, enc_mask_v

    def _get_contextual_target_id(self, input_ids):
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.cpu().tolist()

        # TODO confirm all multi-word targets
        target_ids = list(self.contextual_word_ids.intersection(input_ids))
        # x = ' '.join(self.convert_ids_to_tokens(ids))
        target_id = sorted(target_ids)[-1] # take final subword idx
        index_of_target_id = input_ids.index(target_id)
        return index_of_target_id

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
            esize, pval = scripts.weat_union(X, Y, AX, AY, BX, BY, num_samples,
                                             self.category_X, self.category_Y,
                                             self.category_A, self.category_B)
            results[test_type] = (esize, pval)
        return results

    def run_weat_specific(self, encodings: Dict, num_samples: int):
        results = {}
        for test_type in self.test_types:
            X, Y, AX, AY, BX, BY = self._get_revelant_encodings(test_type, encodings)
            esize, pval = scripts.weat_specific(X, Y, AX, AY, BX, BY, num_samples,
                                                self.category_X, self.category_Y,
                                                self.category_A, self.category_B)
            results[test_type] = (esize, pval)
        return results
        
    def run_weat_intra(self, encodings: Dict, num_samples: int):
        results = {}
        for test_type in self.test_types:
            X, Y, AX, AY, BX, BY = self._get_revelant_encodings(test_type, encodings)
            esize_x, pval_x, esize_y, pval_y =\
                        scripts.weat_intra(X, Y, AX, AY, BX, BY, num_samples,
                                                self.category_X, self.category_Y,
                                                self.category_A, self.category_B)
            results[test_type] = (esize_x, pval_x, esize_y, pval_y)
        return results

    def run_weat_mask(self, encodings: Dict, num_samples: int):
        results = {}
        for mask_type in ['mask_t', 'mask_v']:
            X, Y, AX, AY, BX, BY = self._get_revelant_encodings(mask_type, encodings)
            if len(X) == 0:
                continue
            esize, pval = scripts.weat_union(X, Y, AX, AY, BX, BY, num_samples,
                                             self.category_X, self.category_Y,
                                             self.category_A, self.category_B)
            results[mask_type] = (esize, pval)
        return results
    
    def get_general_vals(self, encodings: Dict, num_samples: int): # TODO rename
        results = {}
        for test_type in self.test_types:
            X, Y, AX, AY, BX, BY = self._get_revelant_encodings(test_type, encodings)
            vals = scripts.get_general_vals(X, Y, AX, AY, BX, BY, n_samples=num_samples)
            results[test_type] = vals
        return results
