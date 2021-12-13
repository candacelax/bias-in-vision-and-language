from collections import defaultdict
from copy import deepcopy
import os
import json
import random
import re
from tqdm import tqdm
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField, ListField, LabelField, SequenceLabelField, ArrayField, MetadataField
from allennlp.data.instance import Instance
#from allennlp.data.tokenizers import Token
#from .box_utils import load_image, resize_image, to_tensor_and_normalize
from .bert_field import BertField, IntArrayField


from allennlp.data.instance import Instance
from allennlp.data.dataset import Batch
from allennlp.data.fields import ListField

from .bert_data_utils import *
from dataloaders.tokenization import BertTokenizer

class BiasDataset(Dataset):
    def __init__(
        self,
        images: Dict[str, List[int]],
        captions: Dict[str, str],
        image_features: Dict[str, torch.FloatTensor],
        image_feature_cap: int,
        bert_model_name: str,
        max_seq_length: int,
        do_lower_case: bool,
        coco_ontology_path: str,
        visual_genome_chunk: bool = False,
        text_only: bool=False,
        expand_coco: bool=False,
        bert_cache: str=None
        ):
        super(BiasDataset, self).__init__()
        self.text_only = text_only
        self.max_seq_length = max_seq_length
        self.expand_coco = expand_coco
        self.expanded = False
        self.is_target_concept = True

        # format annoations
        self.items = self._format_examples(captions, images)
        print(f"{len(self.items)} examples in total.")

        
        print("Loading images...")
        average = 0.0
        self.chunk = {}
        image_screening_parameters = {'image_feature_cap' : 144}
        for image_id in image_features.keys():
            image_feat_variable, image_boxes, confidence  = image_features[image_id]
            if ".npz" not in image_id:
                image_id = image_id + ".npz"

            self.chunk[image_id] = screen_feature(image_feat_variable, image_boxes, confidence, image_screening_parameters)
            average += self.chunk[image_id][2]

        print("{} features on average.".format(average/len(self.chunk)))

        self.tokenizer = BertTokenizer.from_pretrained(
            bert_model_name,
            do_lower_case=do_lower_case,
            cache_dir=bert_cache
            )
        
        with open(os.path.join(coco_ontology_path), 'r') as f:
            coco = json.load(f)
        self.coco_objects = ['__background__'] + [x['name'] for k, x in sorted(coco.items(), key=lambda x: int(x[0]))]
        self.coco_obj_to_ind = {o: i for i, o in enumerate(self.coco_objects)}     

    def num_unique_images(self):
        return self.num_unique_images
        
    def _format_examples(self, captions: Dict, images: Dict):
        items = []
        unique_images = set()
        for image_fp, corresponding_caps in images.items():
            for c in corresponding_caps:
                cap = captions[str(c)]
                items.append({'caption' : cap,
                              'image_id' : image_fp})
                unique_images.add(image_fp)
        self.num_unique_images = len(unique_images)
        return items
            
    def get_image_features_by_training_index(self, index: int):
        item = self.items[index]
        image_file_name = f'{item["image_id"]}.npz'
        return self.chunk[image_file_name]
                
    def __len__(self):
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        sample = {}
        if not self.text_only:
            image_feat_variable, image_boxes, image_dim_variable = self.get_image_features_by_training_index(index)
            image_feat_variable = ArrayField(image_feat_variable)
            image_dim_variable = IntArrayField(np.array(image_dim_variable))
            sample["image_feat_variable"] = image_feat_variable
            sample["image_dim_variable"] = image_dim_variable
            sample["label"] = image_dim_variable
        else:
            sample["label"] = IntArrayField(np.array([0]))

        subword_tokens = self.tokenizer.tokenize(item["caption"])
        bert_example = InputExample(unique_id = index,
                                    text_a = subword_tokens,
                                    text_b = None,
                                    is_correct=None,
                                    max_seq_length = self.max_seq_length)
        
        
        bert_feature = InputFeatures.convert_one_example_to_bias_features(
            example = bert_example,
            tokenizer=self.tokenizer)
        bert_feature.insert_field_into_dict(sample)
        return Instance(sample)

    @staticmethod
    def collate_fn(data):
        if isinstance(data[0], Instance):
            batch = Batch(data)
            td = batch.as_tensor_dict()
            return td
        else:
            images, instances = zip(*data)
            images = torch.stack(images, 0)

            batch = Batch(instances)
            td = batch.as_tensor_dict()
            td['box_mask'] = torch.all(td['boxes'] >= 0, -1).long()
            td['images'] = images
            return td