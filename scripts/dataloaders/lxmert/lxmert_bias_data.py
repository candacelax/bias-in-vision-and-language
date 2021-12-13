# coding=utf-8
# Copyleft 2019 project LXRT.

from collections import defaultdict
import json
import random
from typing import Dict
import numpy as np
import torch
import re
from torch.utils.data import Dataset
from typing import List
from transformers import LxmertTokenizer
from .utils import load_obj_tsv


class InputExample(object):
    """A single training/test example for the language model."""
    def __init__(self, uid, sent, visual_feats=None,
                 obj_labels=None, attr_labels=None,
                 is_matched=None, label=None):
        self.uid = uid
        self.sent = sent
        self.visual_feats = visual_feats
        self.obj_labels = obj_labels
        self.attr_labels = attr_labels
        self.is_matched = is_matched  # whether the visual and obj matched
        self.label = label


class LXMERTBiasDataset:
    def __init__(self, captions: Dict, images: Dict):
        """
        :param splits: The data sources to be loaded
        :param qa_sets: if None, no action
                        o.w., only takes the answers appearing in these dsets
                              and remove all unlabeled data (MSCOCO captions)
        """
        # Loading datasets to data
        self.data = []
        unique_images = set()
        for image_fp, corresponding_caps in images.items():
            for c in corresponding_caps:
                cap = captions[str(c)]
                self.data.append({'caption' : cap,
                                  'image_id' : image_fp.strip(".jpg|.png")})
                unique_images.add(image_fp)
        self.num_unique_images = len(unique_images)
        
    def __len__(self):
        return len(self.data)


def make_uid(img_id, dset, sent_idx):
    return "%s_%s_%03d" % (img_id, dset, sent_idx)


"""
Example in obj tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
"""
class LXMERTBiasTorchDataset(Dataset):
    def __init__(self, bert_model_name: str, dataset: LXMERTBiasDataset, img_data=None):
        super().__init__()
        self.tokenizer = LxmertTokenizer.from_pretrained(bert_model_name)
        self.raw_dataset = dataset
        # Load the dataset
        #if img_data is None:
        #    if feature_filepaths is not None:
        #        img_data = []
        #        [img_data.extend(load_obj_tsv(fp)) for fp in feature_filepaths]
        #    else:
        #        img_data = []
        #        for source in self.raw_dataset.sources:
        #            img_data.extend(load_obj_tsv(Split2ImgFeatPath[source], topk=None))

        self.imgid2img = {}
        for img_datum in img_data:
            self.imgid2img[img_datum['img_id']] = img_datum

        # Filter out the dataset
        used_data = []
        for datum in self.raw_dataset.data:
            if datum["image_id"] in self.imgid2img:
                used_data.append(datum)
            elif datum["image_id"] + '.jpg' in self.imgid2img: # TODO update img ids
                datum["image_id"] = datum["image_id"] + ".jpg"
                used_data.append(datum)
            else: # TODO missing images
                # FUNKY
                raise Exception()
                reps = [
                    ('randmother', 'grandmother'), ('hysics', 'physics'),
                    ('asa', 'nasa'), ('randfather', 'grandfather'),
                    ('ovel', 'novel'), ('oetry', 'poetry')

                ]
                for (orig, new) in reps:
                    if orig in datum["image_id"]:
                        img_id = re.sub(orig, new, datum["image_id"])
                        datum["image_id"] = img_id
                        break
                
                if datum["image_id"] + '.jpg' in self.imgid2img: # TODO update img ids
                    datum["image_id"] = datum["image_id"] + ".jpg"
                    used_data.append(datum)
                else:
                    print(f'missing {datum}')
            
        # Flatten the dataset (into one sent + one image entries)
        self.data = []
        for datum in used_data:
            new_datum = {
                'uid': make_uid(datum['image_id'], "bias", 0),
                'img_id': datum["image_id"],
                'sent': datum["caption"]
            }
            self.data.append(new_datum)

        print("Use %d data in torch dataset" % (len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]

        uid = datum['uid']
        img_id = datum['img_id']

        # Get image info
        img_info = self.imgid2img[img_id]
        obj_num = img_info['num_boxes']
        feats = img_info['features'].copy()
        boxes = img_info['boxes'].copy()
        obj_labels = img_info['objects_id'].copy()
        obj_confs = None # img_info['objects_conf'].copy() if 'objects_conf' in img_info is not None else None
        attr_labels = None #img_info['attrs_id'].copy() if 'attrs_id' in img_info is not None else None
        attr_confs = None # img_info['attrs_conf'].copy() if 'attrs_conf' in img_info is not None else None
        assert obj_num == len(boxes) == len(feats)

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)

        sent = datum['sent']
        label = None
        
        # Create target
        example = InputExample(
            uid, sent, (feats, boxes),
            (obj_labels, obj_confs), (attr_labels, attr_confs),
            False, label
        )
        return example

    def collate_fn(self, data):
        batch_sents, batch_image_feats, batch_image_boxes, batch_obj_indices = [], [], [], []
        for example in data:
            batch_sents.append(example.sent)
            feat, boxes = example.visual_feats
            obj_indices, _ = example.obj_labels # these are really indices to labels
            batch_image_feats.append(torch.tensor(feat))
            batch_image_boxes.append(torch.tensor(boxes))
            batch_obj_indices.append(torch.tensor(obj_indices))
        
        batch_out = self.tokenizer(
            batch_sents,
            padding=True,
            truncation=True,
            return_tensors='pt'
            )
        batch_out['visual_feats'] = torch.stack(batch_image_feats)
        batch_out['visual_pos'] = torch.stack(batch_image_boxes)
        batch_out['obj_indices'] = torch.stack(batch_obj_indices)
        batch_out = {k:v.cuda() for k,v in batch_out.items()}
        return batch_out