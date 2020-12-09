from attrdict import AttrDict
import numpy as np
import os
from os import path
import re
from typing import Any, Dict, List, Iterable
import torch
from torch.utils.data import DataLoader, Dataset

MODEL2LOADER = {
    'visualbert' : 'VisualBERTDataLoader',
    'vilbert' : 'ViLBERTDataLoader',
    'lxmert' : 'LXMERTDataLoader',
    'vlbert' : 'VLBERTDataLoader'
    }

class BiasDataLoader(DataLoader):
    def __init__(
        self,
        params: AttrDict,
        category: str,
        images: Dict[str, List[int]],
        captions: Dict[str, str],
        contextual_words: List[str],
        image_features: Dict[str, Any],
        mask_token: str='[MASK]',
        pad_token: str='[PAD]'
        ):

        dataset = self.load_dataset(params, images, captions, image_features)
        collate_fn = getattr(dataset, 'collate_fn', None)
        
        super().__init__(
            dataset=dataset,
            batch_size=params.batch_size // params.num_gpus,
            shuffle=False,
            num_workers=0, # TODO remove meparams.get('num_workers', 8),
            collate_fn=collate_fn,
            drop_last=False,
            pin_memory=False
        )
        
        self.tokenizer = dataset.tokenizer
        self.category = category
        self.contextual_words = contextual_words
        self.contextual_words_with_people = contextual_words + ['people', 'person', 'woman', 'women', 'man', 'men']
        
        self.convert_tokens_to_ids = dataset.tokenizer.convert_tokens_to_ids
        self.convert_ids_to_tokens = dataset.tokenizer.convert_ids_to_tokens
        self.mask_token_id = dataset.tokenizer.convert_tokens_to_ids([mask_token])[0]
        self.pad_token_id = dataset.tokenizer.convert_tokens_to_ids([pad_token])[0]

        self.contextual_word_ids = [self.convert_tokens_to_ids(dataset.tokenizer.tokenize(word)) for word in self.contextual_words]
        
        # we'll find the longest matching tokenized spans first when we replace with masked id
        self.contextual_word_ids_as_strings = []
        for cwids in self.contextual_word_ids:
            cwids = ' '.join([str(c) for c in cwids])
            self.contextual_word_ids_as_strings.append(cwids)
        self.contextual_word_ids_as_strings.sort(key=lambda x: len(x), reverse=True)

    def mask_contextual_words_in_batch(self, batch: Dict, input_id_key: str):
        # find matching spans of contextual word ids in input ids
        # then replace with mask_id
        # and possibly add padding tokens for multi-token id contextual words
        max_len = batch[input_id_key].shape[1]
        for idx, input_ids in enumerate(batch[input_id_key]):
            # convert to string and replace matching spans
            input_tokens = None
            input_ids_as_str = ' '.join([str(i) for i in input_ids.tolist()])
            for cws in self.contextual_word_ids_as_strings:
                if cws in input_ids_as_str:
                    input_ids_as_str = re.sub(cws, str(self.mask_token_id), input_ids_as_str)
                    break

            assert str(self.mask_token_id) in input_ids_as_str, \
                f'Nothing masked!\nInput tokens: {input_ids} {input_ids_as_str} {input_tokens} \n {self.contextual_word_ids_as_strings}'
            
            # convert back to list
            input_ids = [int(t) for t in input_ids_as_str.split(' ')]

            # check if we need to add padding
            if len(input_ids) != max_len:
                input_ids += [self.pad_token_id] * (max_len - len(input_ids))
            input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

            # update batch
            batch[input_id_key][idx] = torch.tensor(input_ids, device=batch[input_id_key].device)
        return batch    
        

class VisualBERTDataLoader(BiasDataLoader):
    def load_dataset(
        self,
        params: Dict[str, Any],
        images: Dict[str, List[int]],
        captions: Dict[str, str],
        image_features: Dict[str, Any]
    ):
        from dataloaders.visualbert.bias_dataset import BiasDataset
        return BiasDataset(
            images=images,
            captions=captions,
            image_features=image_features,
            coco_ontology_path=params.coco_ontology,
            image_feature_cap=params.image_feature_cap,
            bert_model_name=params.bert_model_name,
            max_seq_length=params.max_seq_length,
            do_lower_case=params.do_lower_case,
            bert_cache=params.bert_cache
            )

    @staticmethod
    def load_image_features(filepath: str):
        return torch.load(filepath)

class ViLBERTDataLoader(BiasDataLoader):
    def load_dataset(
        self,
        params: Dict[str, Any],
        images: Dict[str, List[int]],
        captions: Dict[str, str],
        image_features: Dict[str, Any]
    ):
        from dataloaders.vilbert.bias_dataset import BiasDataset
        obj_list = ['background']
        with open(params.path_to_obj_list) as f:
            [obj_list.append(line.strip()) for line in f]
        return BiasDataset(
            bert_model_name=params.bert_model_name,
            captions=captions,
            images=images,
            dataset_type=params.dataset_type,
            image_features=image_features,
            obj_list=obj_list,
            seq_len=params.max_seq_length
            )

    @staticmethod
    def load_image_features(filepath: str):
        from dataloaders.vilbert._image_features_reader import ImageFeaturesH5ReaderWithObjClasses
        return ImageFeaturesH5ReaderWithObjClasses(filepath)

class LXMERTDataLoader(BiasDataLoader):
    def load_dataset(
        self,
        params: Dict[str, Any],
        images: Dict[str, List[int]],
        captions: Dict[str, str],
        image_features: Dict[str, Any]
    ):
        from transformers import LxmertTokenizer
        from dataloaders.lxmert.lxmert_bias_data import (
            LXMERTBiasDataset,
            LXMERTBiasTorchDataset
            )
        
        # these objects correspond to image region labels that can be masked
        self.obj_list = ['background']
        with open(params.path_to_obj_list) as f:
            [self.obj_list.append(line.strip()) for line in f]
        return LXMERTBiasTorchDataset(
            bert_model_name=params.bert_model_name,
            dataset=LXMERTBiasDataset(
                    images=images,
                    captions=captions,
                    ),
            img_data=image_features
            )

    @staticmethod
    def load_image_features(image_features_fp: str):
        from dataloaders.lxmert.utils import load_obj_tsv
        if os.path.exists(image_features_fp):
            return load_obj_tsv(image_features_fp)
        else:
            img_data = []
            basedir = path.dirname(image_features_fp)
            for f in os.listdir(basedir):
                if re.match(f'{image_features_fp}.*', path.join(basedir,f)):
                    img_data.extend(load_obj_tsv(path.join(basedir, f)))
            return img_data

    def mask_image_regions(self, batch: Dict, obj_indices: torch.Tensor):
        num_examples, num_indices = obj_indices.shape
        feat_dim = batch['visual_feats'].shape[-1]
        for example_idx in range(num_examples):
            for i,obj_idx in enumerate(obj_indices[example_idx].tolist()):
                if self.obj_list[obj_idx] in self.contextual_words_with_people:
                    batch['visual_feats'][example_idx][i] = torch.zeros(feat_dim)
        return batch

class VLBERTDataLoader(BiasDataLoader):
    def load_dataset(
        self,
        params: Dict[str, Any],
        images: Dict[str, List[int]],
        captions: Dict[str, str],
        image_features: Dict[str, Any]
    ):  
        from dataloaders.vlbert import BiasDataset

        # these objects correspond to image region labels that can be masked
        self.obj_list = ['background']
        with open(params.path_to_obj_list) as f:
            [self.obj_list.append(line.strip()) for line in f]

        self.transform = lambda img, shape: resize(img, shape)
        return BiasDataset(
            #**params.model_config,
            image_features=image_features,
            cache_dir=params.bert_cache,
            captions=captions,
            images=images
            )
    
    @staticmethod
    def load_image_features(feature_dir: str):
        from dataloaders.vlbert import BiasDataset
        #tsv_names = ['image_id', 'image_h', 'image_w', 'num_boxes', 'boxes', 'features', 'cls_prob', 'classes']
        imgid2features = {}
        for fp in os.listdir(feature_dir):
            full_path = os.path.join(feature_dir, fp)
            if not re.match(full_path, '.*tsv') and not os.path.isfile(full_path):
                continue
            
            with open(os.path.join(feature_dir, fp)) as f:
                for line in f:
                    feature_dict = {k:v for k,v in zip(BiasDataset.tsv_names, line.strip().split('\t'))}
                    img_id = feature_dict['image_id']
                    imgid2features[img_id] = feature_dict
        return imgid2features


    def mask_input_ids(self, input_ids: torch.Tensor):
        batch_out = self.mask_contextual_words_in_batch({'input_ids' : input_ids}, 'input_ids')
        return batch_out['input_ids']

    def mask_input_features(self, boxes: torch.Tensor, object_labels: torch.Tensor):
        for example_idx, labels in enumerate(object_labels[:len(boxes)]):
            for label_idx, label in enumerate(labels):
                if self.obj_list[label] in self.contextual_words_with_people:
                    if label_idx >= len(boxes[example_idx]):
                        continue
                    boxes[example_idx][label_idx] = torch.zeros_like(boxes[example_idx][label_idx])
        return boxes
