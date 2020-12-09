import random
import os
import time
import json
import jsonlines
from PIL import Image
import base64
import numpy as np
import logging
import re

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from models.vlbert.external.pytorch_pretrained_bert import BertTokenizer

from models.vlbert.common.utils.zipreader import ZipReader
from models.vlbert.common.utils.create_logger import makedirsExist


class BiasDataset(Dataset):
    tsv_names = ['image_id', 'image_h', 'image_w', 'num_boxes', 'boxes', 'features', 'cls_prob', 'classes']

    def __init__(self,
                cache_dir,
                image_features,
                captions,
                images,
                image_set=None, root_path=None, data_path=None, seq_len=64,
                with_precomputed_visual_feat=True,
                transform=None, test_mode=False,
                zip_mode=False, cache_mode=False, cache_db=False, ignore_db_cache=True,
                tokenizer=None, pretrained_model_name=None,
                add_image_as_a_box=False,
                aspect_grouping=False, **kwargs):
        """
        Conceptual Captions Dataset

        :param ann_file: annotation jsonl file
        :param image_set: image folder name, e.g., 'vcr1images'
        :param root_path: root path to cache database loaded from annotation file
        :param data_path: path to vcr dataset
        :param transform: transform
        :param test_mode: test mode means no labels available
        :param zip_mode: reading images and metadata in zip archive
        :param cache_mode: cache whole dataset to RAM first, then __getitem__ read them from RAM
        :param ignore_db_cache: ignore previous cached database, reload it from annotation file
        :param tokenizer: default is BertTokenizer from pytorch_pretrained_bert
        :param add_image_as_a_box: add whole image as a box
        :param aspect_grouping: whether to group images via their aspect
        :param kwargs:
        """
        super(BiasDataset, self).__init__()
        assert not cache_mode, 'currently not support cache mode!'
        assert not test_mode

        self.image_features = image_features
        self.seq_len = seq_len
        self.data_path = data_path
        self.root_path = root_path
        self.with_precomputed_visual_feat = with_precomputed_visual_feat
        self.image_set = image_set
        self.transform = transform
        self.test_mode = test_mode
        self.zip_mode = zip_mode
        self.cache_mode = cache_mode
        self.cache_db = cache_db
        self.ignore_db_cache = ignore_db_cache
        self.aspect_grouping = aspect_grouping
        self.cache_dir = cache_dir
        self.add_image_as_a_box = add_image_as_a_box
        if not os.path.exists(self.cache_dir):
            makedirsExist(self.cache_dir)
        self.tokenizer = tokenizer if tokenizer is not None \
            else BertTokenizer.from_pretrained(
            'bert-base-uncased' if pretrained_model_name is None else pretrained_model_name,
            cache_dir=self.cache_dir)
        
        self.examples = self._format_examples(captions, images)

        if self.aspect_grouping:
            assert False, "not support aspect grouping currently!"
            self.group_ids = self.group_aspect(self.database)

    def _format_examples(self, captions, images):
        items = []
        unique_images = set()
        for image_fp, corresponding_caps in images.items():
            for c in corresponding_caps:
                cap = captions[str(c)]
                items.append(
                    {'caption' : cap,
                    'image_id' : image_fp}
                    )
                unique_images.add(image_fp)
        self.num_unique_images = len(unique_images)
        return items

    @property
    def data_names(self):
        return [
            'image', 'boxes', 'im_info', 'text', 'relationship_label', 'mlm_labels', 'mvrc_ops', 'mvrc_labels', 'object_labels'
            ]

    def __getitem__(self, index):
        # image data
        example = self.examples[index]
        frcnn_data = self.image_features[example['image_id']]
        boxes = np.frombuffer(base64.b64decode(frcnn_data['boxes']),
                              dtype=np.float32).reshape((int(frcnn_data['num_boxes']), -1))
        boxes_cls_scores = np.frombuffer(base64.b64decode(frcnn_data['classes']),
                                         dtype=np.float32).reshape((int(frcnn_data['num_boxes']), -1))
        
        boxes_max_conf = boxes_cls_scores.max(axis=1)
        inds = np.argsort(boxes_max_conf)[::-1]
        boxes = boxes[inds]
        boxes_cls_scores = boxes_cls_scores[inds]
        boxes = torch.as_tensor(boxes)

        #
        image = None
        w0, h0 = float(frcnn_data['image_w']), float(frcnn_data['image_h'])
        boxes_features = np.frombuffer(self.b64_decode(frcnn_data['features']),
                                    dtype=np.float32).reshape((int(frcnn_data['num_boxes']), -1))
        boxes_features = boxes_features[inds]
        boxes_features = torch.as_tensor(boxes_features)

        if self.add_image_as_a_box:
            image_box = torch.as_tensor([[0.0, 0.0, w0 - 1.0, h0 - 1.0]])
            boxes = torch.cat((image_box, boxes), dim=0)
            if self.with_precomputed_visual_feat:
                image_box_feat = boxes_features.mean(dim=0, keepdim=True)
                boxes_features = torch.cat((image_box_feat, boxes_features), dim=0)

        # no transform because we're loading features directly
        im_info = torch.tensor([w0, h0, 1.0, 1.0, index])

        if image is None and (not self.with_precomputed_visual_feat):
            w = int(im_info[0].item())
            h = int(im_info[1].item())
            image = im_info.new_zeros((3, h, w), dtype=torch.float)

        # clamp boxes
        w = im_info[0].item()
        h = im_info[1].item()
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=w-1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=h-1)

        caption_tokens = self.tokenizer.tokenize(example['caption'])
        text_tokens = ['[CLS]'] + caption_tokens + ['[SEP]']
        text_ids = self.tokenizer.convert_tokens_to_ids(text_tokens)
        mlm_labels = [1] * len(text_ids)

        if self.with_precomputed_visual_feat:
            boxes = torch.cat((boxes, boxes_features), dim=1)

        # set defaults for no-image
        relationship_label = 1
        mvrc_ops = [0] * boxes.shape[0]
        mvrc_labels = [np.zeros_like(boxes_cls_scores[0])] * boxes.shape[0]
        object_labels = torch.tensor(np.frombuffer(base64.b64decode(frcnn_data['classes'])).astype(np.int32))

        # truncate seq to max len
        if len(text_ids) + len(boxes) > self.seq_len:
            text_len_keep = len(text_ids)
            box_len_keep = len(boxes)
            while (text_len_keep + box_len_keep) > self.seq_len and (text_len_keep > 0) and (box_len_keep > 0):
                if box_len_keep > text_len_keep:
                    box_len_keep -= 1
                else:
                    text_len_keep -= 1
            if text_len_keep < 2:
                text_len_keep = 2
            if box_len_keep < 1:
                box_len_keep = 1
            boxes = boxes[:box_len_keep]
            text_ids = text_ids[:(text_len_keep - 1)] + [text_ids[-1]]
            mlm_labels = mlm_labels[:(text_len_keep - 1)] + [mlm_labels[-1]]
            mvrc_ops = mvrc_ops[:box_len_keep]
            mvrc_labels = mvrc_labels[:box_len_keep]
        return image, boxes, im_info, text_ids, relationship_label, mlm_labels, mvrc_ops, mvrc_labels, object_labels

    def random_mask_region(self, regions_cls_scores):
        num_regions, num_classes = regions_cls_scores.shape
        output_op = []
        output_label = []
        for k, cls_scores in enumerate(regions_cls_scores):
            prob = random.random()
            # mask region with 15% probability
            if prob < 0.15:
                prob /= 0.15

                if prob < 0.9:
                    # 90% randomly replace appearance feature by "MASK"
                    output_op.append(1)
                else:
                    # -> rest 10% randomly keep current appearance feature
                    output_op.append(0)

                # append class of region to output (we will predict these later)
                output_label.append(cls_scores)
            else:
                # no masking region (will be ignored by loss function later)
                output_op.append(0)
                output_label.append(np.zeros_like(cls_scores))

        # # if no region masked, random choose a region to mask
        # if all([op == 0 for op in output_op]):
        #     choosed = random.randrange(0, len(output_op))
        #     output_op[choosed] = 1
        #     output_label[choosed] = regions_cls_scores[choosed]

        return output_op, output_label

    @staticmethod
    def b64_decode(string):
        return base64.decodebytes(string.encode())

    @staticmethod
    def group_aspect(database):
        print('grouping aspect...')
        t = time.time()

        # get shape of all images
        widths = torch.as_tensor([idb['width'] for idb in database])
        heights = torch.as_tensor([idb['height'] for idb in database])

        # group
        group_ids = torch.zeros(len(database))
        horz = widths >= heights
        vert = 1 - horz
        group_ids[horz] = 0
        group_ids[vert] = 1

        print('Done (t={:.2f}s)'.format(time.time() - t))

        return group_ids

    def __len__(self):
        return len(self.examples)

    def _load_image(self, path):
        if '.zip@' in path:
            return self.zipreader.imread(path).convert('RGB')
        else:
            return Image.open(path).convert('RGB')

    def _load_json(self, path):
        if '.zip@' in path:
            f = self.zipreader.read(path)
            return json.loads(f.decode())
        else:
            return json.load(f.decode())

    def collate_fn(self, batch_in):
        batch_size = len(batch_in)
        batch_out = []
        max_h, max_w = None,None
        for idx, (name,val) in enumerate(zip(self.data_names, zip(*batch_in))):
            if name == 'im_info':
                batch_out.append(torch.stack(val))
            elif name == 'image':
                if val[0] is not None:
                    max_h = max([v.shape[1] for v in val])
                    max_w = max([v.shape[2] for v in val])
                    out_val = torch.zeros((batch_size, 3, max_h, max_w)) # images are all zero bc features are precomputed
                    batch_out.append(out_val)
                else:
                    batch_out.append(val)
            elif name == 'boxes':
                padded = pad_sequence(val, batch_first=True)
                batch_out.append(padded)
            elif isinstance(val[0], torch.Tensor):
                out_val = pad_sequence(val, batch_first=True)
                batch_out.append(out_val)
            elif isinstance(val[0], list):
                val = [torch.tensor(v) for v in val]
                out_val = pad_sequence(val, batch_first=True)
                batch_out.append(out_val)
            elif isinstance(val[0], int):
                batch_out.append(torch.Tensor(val))
            else:
                raise Exception(f'{name} not added')
        return batch_out
