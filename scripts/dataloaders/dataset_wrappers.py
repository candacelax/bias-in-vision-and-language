
from attrdict import AttrDict
import os
from os import path
import re
import torch
from typing import Any, Callable, Dict, List
from torch.utils.data import Dataset

class VisualBERTDatasetWrapper:
    def __init__(
        self,
        params: Dict[str, Any],
        images: Dict[str, List[int]],
        captions: Dict[str, str],
        image_features_path_or_dir: Dict[str, Any],
        **kwargs
    ):
        assert len(kwargs) == 0, "No kwargs should be passed for VisualBERT"
        from .visualbert.bias_dataset import BiasDataset
        image_features = torch.load(image_features_path_or_dir)
        self.dataset = BiasDataset(
            images=images,
            captions=captions,
            image_features=image_features,
            coco_ontology_path=params.coco_ontology,
            bert_model_name=params.bert_model_name,
            max_seq_length=params.max_seq_length,
            do_lower_case=params.do_lower_case,
            bert_cache=params.bert_cache
            )

class ViLBERTDatasetWrapper:
    def __init__(
        self,
        params: Dict[str, Any],
        images: Dict[str, List[int]],
        captions: Dict[str, str],
        image_features_path_or_dir: Dict[str, Any],
        **kwargs
        
    ):
        from .vilbert.bias_dataset import BiasDataset
        from .vilbert._image_features_reader import ImageFeaturesH5ReaderWithObjClasses
        assert len(kwargs) == 0, "No kwargs should be passed for ViLBERT"

        obj_list = ['background']
        with open(params.path_to_obj_list) as f:
            [obj_list.append(line.strip()) for line in f]

        image_features = ImageFeaturesH5ReaderWithObjClasses(image_features_path_or_dir)
        self.dataset = BiasDataset(
            bert_model_name=params.bert_model_name,
            captions=captions,
            images=images,
            dataset_type=params.dataset_type,
            image_features=image_features,
            obj_list=obj_list,
            seq_len=params.max_seq_length
            )

class LXMERTDatasetWrapper:
    def __init__(
        self,
        params: Dict[str, Any],
        images: Dict[str, List[int]],
        captions: Dict[str, str],
        image_features_path_or_dir: Dict[str, Any],
        **kwargs
    ):
        from .lxmert.lxmert_bias_data import (
            LXMERTBiasDataset, LXMERTBiasTorchDataset
        )

        assert len(kwargs) == 0, "No kwargs should be passed for LXMERT"

        # these objects correspond to image region labels that can be masked
        self.obj_list = ['background']
        with open(params.path_to_obj_list) as f:
            [self.obj_list.append(line.strip()) for line in f]
        
        image_features = self.load_image_features(image_features_path_or_dir)

        self.dataset = LXMERTBiasTorchDataset(
            bert_model_name=params.bert_model_name,
            dataset=LXMERTBiasDataset(
                        images=images,
                        captions=captions,
                    ),
            img_data=image_features
            )

    @staticmethod
    def load_image_features(image_features_path_or_dir: str):
        from .lxmert.utils import load_obj_tsv
        if path.exists(image_features_path_or_dir):
            return load_obj_tsv(image_features_path_or_dir)
        else:
            img_data = []
            basedir = path.dirname(image_features_path_or_dir)
            for f in os.listdir(basedir):
                if re.match(f'{image_features_path_or_dir}.*', path.join(basedir,f)):
                    img_data.extend(load_obj_tsv(path.join(basedir, f)))
            return img_data

    def mask_image_regions(self, batch: Dict, obj_indices: torch.Tensor):
        num_examples, _ = obj_indices.shape
        feat_dim = batch['visual_feats'].shape[-1]
        for example_idx in range(num_examples):
            for i,obj_idx in enumerate(obj_indices[example_idx].tolist()):
                if self.obj_list[obj_idx] in self.contextual_words_with_people:
                    batch['visual_feats'][example_idx][i] = torch.zeros(feat_dim)
        return batch

class VLBERTDatasetWrapper:
    def __init__(
        self,
        params: Dict[str, Any],
        images: Dict[str, List[int]],
        captions: Dict[str, str],
        image_features_path_or_dir: Dict[str, Any],
        **kwargs
    ):
        from .vlbert import VLBERTBiasDataset

        assert len(kwargs) == 0, "No kwargs should be passed for VLBERT"

        # these objects correspond to image region labels that can be masked
        self.obj_list = ['background']
        with open(params.path_to_obj_list) as f:
            [self.obj_list.append(line.strip()) for line in f]

        image_features = self.load_image_features(image_features_path_or_dir)
        self.transform = lambda img, shape: resize(img, shape)
        self.dataset = VLBERTBiasDataset(
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

class CustomModelDatasetWrapper(Dataset):
    def __init__(
        self,
        params: Dict[str, Any],
        images: Dict[str, List[int]],
        captions: Dict[str, str],
        image_features_path_or_dir: Dict[str, Any],
        load_or_compute_image_features: Callable,
        create_dataset: Callable
    ):
        assert load_or_compute_image_features is not None
        assert create_dataset is not None

        image_features = load_or_compute_image_features(
            images, captions, image_features_path_or_dir
            )
        self.dataset = create_dataset(
            params, images, captions, image_features
            )


DATASET_CLASS = {
    'visualbert' : VisualBERTDatasetWrapper,
    'vilbert' : ViLBERTDatasetWrapper,
    'lxmert' : LXMERTDatasetWrapper,
    'vlbert' : VLBERTDatasetWrapper,
    'custom' : CustomModelDatasetWrapper
    }

def create_dataset(
        params: AttrDict,
        captions: Dict[str, str],
        images: Dict[str, List[int]],
        image_features_path_or_dir,
        **kwargs
    ):
    dataset_wrapper = DATASET_CLASS[params.model_type](
        params=params,
        captions=captions,
        images=images,
        image_features_path_or_dir=image_features_path_or_dir,
        **kwargs
    )
    return dataset_wrapper