import tensorpack.dataflow as td
import torch
from torch.utils.data import Dataset

from ..tokenization import BertTokenizer
import re

def iou(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (K, 4) ndarray of float
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = anchors.size(0)
    K = gt_boxes.size(0)

    gt_boxes_area = ((gt_boxes[:,2] - gt_boxes[:,0] + 1) *
                (gt_boxes[:,3] - gt_boxes[:,1] + 1)).view(1, K)

    anchors_area = ((anchors[:,2] - anchors[:,0] + 1) *
                (anchors[:,3] - anchors[:,1] + 1)).view(N, 1)

    boxes = anchors.view(N, 1, 4).expand(N, K, 4)
    query_boxes = gt_boxes.view(1, K, 4).expand(N, K, 4)

    iw = (torch.min(boxes[:,:,2], query_boxes[:,:,2]) -
        torch.max(boxes[:,:,0], query_boxes[:,:,0]) + 1)
    iw[iw < 0] = 0

    ih = (torch.min(boxes[:,:,3], query_boxes[:,:,3]) -
        torch.max(boxes[:,:,1], query_boxes[:,:,1]) + 1)
    ih[ih < 0] = 0

    ua = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / ua

    return overlaps

def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)

class InputExample(object):
    """A single training/test example for the language model."""

    def __init__(
        self, image_feat=None, image_target=None, caption=None, lm_labels=None, image_loc=None,
            num_boxes=None, cls_indices=None
    ):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            tokens_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            tokens_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.image_feat = image_feat
        self.caption = caption
        self.lm_labels = lm_labels  # masked words for language model
        self.image_loc = image_loc
        self.image_target = image_target
        self.num_boxes = num_boxes
        self.cls_indices = cls_indices

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
        self,
        input_ids=None,
        input_mask=None,
        segment_ids=None,
        lm_label_ids=None,
        image_feat=None,
        image_loc=None,
        image_label=None,
        image_mask=None,
        coattention_mask=None,
        masked_image_feat=None,
        masked_image_label=None
    ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.lm_label_ids = lm_label_ids
        self.image_feat = image_feat
        self.image_loc = image_loc
        self.image_label = image_label
        self.image_mask = image_mask
        self.coattention_mask = coattention_mask
        self.masked_image_feat = masked_image_feat
        self.masked_image_label = masked_image_label
    
class BiasDataset(Dataset):
    def __init__(
            self,
            bert_model_name,
            captions,
            images,
            dataset_type,
            image_features,
            obj_list,
            seq_len,
            encoding="utf-8"
    ):
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=True)
        self.dataset_type = dataset_type
        self.imageid2filepath = {}

        entries = self._load_entries(images, captions, dataset_type)
        
        preprocess_function = BertPreprocessBatch(
            tokenizer=self.tokenizer,
            obj_list=obj_list,
            seq_len=seq_len,
            region_len=36,
            image_features=image_features,
            imageid2filepath=self.imageid2filepath,
            encoding=encoding,
            predict_feature=False
        )
        ds = td.MapData(entries, preprocess_function)
        self.formatted_entries = list(ds.get_data())

    def _load_entries(self, images, captions, dataset_type):
        entries = []
        for image_fp, corresponding_caps in images.items():
            for c in corresponding_caps:
                cap = captions[str(c)]
                if dataset_type == 'coco':
                    image_id = int(re.sub('^0*', '', image_fp.strip('.jpg|.png').split('_')[-1]))
                    self.imageid2filepath[image_id] = image_fp
                elif dataset_type == 'concap':
                    image_id = int(image_fp)
                    self.imageid2filepath[image_id] = image_fp
                else:
                    image_id = len(entries)
                    self.imageid2filepath[image_id] = image_fp
                entries.append({'caption' : cap, 'image_id' : image_id})
        return entries

    def __getitem__(self, idx: int):
        return self.formatted_entries[idx]

    def collate_fn(self, data):
        batch = []
        for i,vals in enumerate(zip(*data)):
            if isinstance(vals[0], torch.Tensor):
                vals = torch.stack(vals)
            else:
                vals = torch.tensor(vals)
            batch.append(vals)
        
        tensorized_batch = [torch.tensor(x) for x in batch]
        input_ids, input_mask, segment_ids, lm_label_ids, image_feat, \
            image_loc, image_label, image_mask, image_id, coattention_mask,\
            masked_image_feat, masked_image_label = batch

        batch_size = input_ids.shape[0]

        g_image_feat = torch.sum(image_feat, axis=1) / torch.sum(image_mask, axis=1, keepdims=True)
        image_feat = torch.cat((g_image_feat.unsqueeze(1), image_feat), dim=1)#.type(torch.float32)

        #g_image_loc = torch.tensor([[0,0,1,1,1]], dtype=torch.float32).repeat(batch_size,1)
        g_image_loc = torch.tensor([[0,0,1,1,1]], dtype=image_loc.dtype).repeat(batch_size,1)
        image_loc = torch.cat((g_image_loc.unsqueeze(1), image_loc), dim=1)#.type(torch.float32)
        
        g_image_mask = torch.ones((batch_size, 1), dtype=image_mask.dtype)  #np.repeat(np.array([[1]]), batch_size, axis=0)
        image_mask = torch.cat((g_image_mask, image_mask), dim=1) #np.concatenate([g_image_mask, image_mask], axis=1)

        masked_g_image_feat = torch.sum(masked_image_feat, axis=1) / \
                                torch.sum(image_mask, axis=1, keepdims=True)
        masked_image_feat = torch.cat((masked_g_image_feat.unsqueeze(1), masked_image_feat), dim=1) #[np.expand_dims(masked_g_image_feat, axis=1), masked_image_feat], axis=1)
        
        a = (input_ids, input_mask, segment_ids, lm_label_ids, image_feat, \
                    image_loc, image_label, image_mask, image_id, coattention_mask,
                    masked_image_feat, masked_image_label)

        return (input_ids, input_mask, segment_ids, lm_label_ids, image_feat, \
                    image_loc, image_label, image_mask, image_id, coattention_mask,
                    masked_image_feat, masked_image_label)

    def __len__(self):
        return len(self.formatted_entries)


class BertPreprocessBatch(object):
    def __init__(
        self,
        tokenizer,
        obj_list,
        seq_len,
        region_len,
        image_features,
        imageid2filepath,
        encoding="utf-8",
        predict_feature=False
    ):
        self.seq_len = seq_len
        self.region_len = region_len
        self.tokenizer = tokenizer
        self.obj_list = obj_list
        self.image_features = image_features
        self.imageid2filepath = imageid2filepath
        self.predict_feature = predict_feature

    def __call__(self, data):
        caption = data['caption']
        image_id = data['image_id']
        
        image_fp = self.imageid2filepath[image_id]
        image_feature, num_boxes, image_location, image_location_ori, cls_indices = self.image_features[image_fp]

        num_boxes = min(self.region_len, num_boxes) # TODO
        image_feature = image_feature[:self.region_len] #torch.tensor(image_feature[:self.region_len], dtype=torch.float32)
        image_location = image_location[:self.region_len] #torch.tensor(image_location[:self.region_len], dtype=torch.float32)
            
        tokens_caption = self.tokenizer.tokenize(caption)
        cur_example = InputExample(
            image_feat=image_feature,
            caption=tokens_caption,
            image_loc=image_location,
            num_boxes=num_boxes,
            cls_indices=cls_indices
            )
        
        # transform sample to features
        cur_features = self.convert_example_to_features(
            cur_example,
            self.seq_len,
            self.tokenizer,
            self.obj_list,
            self.region_len
            )
        cur_tensors = (
            cur_features.input_ids,
            cur_features.input_mask,
            cur_features.segment_ids,
            cur_features.lm_label_ids,
            cur_features.image_feat,
            cur_features.image_loc,
            cur_features.image_label,
            cur_features.image_mask,
            image_id,
            cur_features.coattention_mask,
            cur_features.masked_image_feat,
            cur_features.masked_image_label
        )
        return cur_tensors
        
    def convert_example_to_features(self, example, max_seq_length, tokenizer, obj_list, max_region_length):
        """
        Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
        IDs, LM labels, input_mask, CLS and SEP tokens etc.
        :param example: InputExample, containing sentence input as strings and is_next label
        :param max_seq_length: int, maximum length of sequence.
        :param tokenizer: Tokenizer
        :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
        """
        image_feat = example.image_feat
        caption = example.caption
        image_loc = example.image_loc
        image_target = example.image_target
        num_boxes = int(example.num_boxes)
        cls_indices = example.cls_indices
        self._truncate_seq_pair(caption, max_seq_length - 2)
        caption_label = self.label_caption(caption, tokenizer)
        image_label = [-1] * len(image_feat)
        masked_image_feat, masked_image_label =\
                self.mask_region(image_feat, num_boxes, cls_indices, obj_list)

        # concatenate lm labels and account for CLS, SEP, SEP
        # lm_label_ids = ([-1] + caption_label + [-1] + image_label + [-1])
        lm_label_ids = [-1] + caption_label + [-1]
        # image_label = ([-1] + image_label)

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []

        tokens.append("[CLS]")
        segment_ids.append(0)
        # for i in range(36):
        #     # tokens.append(0)
        #     segment_ids.append(0)

        # tokens.append("[SEP]")
        # segment_ids.append(0)
        for token in caption:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        # input_ids = input_ids[:1] input_ids[1:]
        input_mask = [1] * (len(input_ids))
        image_mask = [1] * (num_boxes)
        # Zero-pad up to the visual sequence length.
        while len(image_mask) < max_region_length:
            image_mask.append(0)
            image_label.append(-1)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            lm_label_ids.append(-1)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(lm_label_ids) == max_seq_length
        assert len(image_mask) == max_region_length
        assert len(image_label) == max_region_length

        coattention_mask = torch.zeros((max_region_length, max_seq_length))
        features = InputFeatures(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            input_mask=torch.tensor(input_mask, dtype=torch.long),
            segment_ids=torch.tensor(segment_ids, dtype=torch.long),
            lm_label_ids=torch.tensor(lm_label_ids, dtype=torch.long),
            image_feat=torch.tensor(image_feat, dtype=torch.float),
            image_loc=torch.tensor(image_loc, dtype=torch.float),
            image_label=torch.tensor(image_label, dtype=torch.long),
            image_mask=torch.tensor(image_mask, dtype=torch.long),
            coattention_mask=torch.tensor(coattention_mask, dtype=torch.long),
            masked_image_feat=torch.tensor(masked_image_feat, dtype=torch.float),
            masked_image_label=torch.tensor(masked_image_label, dtype=torch.long),
        )
        return features

    def _truncate_seq_pair(self, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_b)
            if total_length <= max_length:
                break

            tokens_b.pop()

    def label_caption(self, tokens, tokenizer):
        output_label = []
        for token in tokens:
            try:
                output_label.append(tokenizer.vocab[token])
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_label.append(tokenizer.vocab["[UNK]"])
                warning.warn(
                    "Cannot find token '{}' in vocab. Using [UNK] insetad".format(token)
                )
        return output_label

    def mask_region(self, image_feat, num_boxes, cls_indices, obj_list):        
        """
        """
        output_label = []
        for i in range(num_boxes):
            cls_idx = cls_indices[i]
            cls = obj_list[cls_idx]
            
            if cls == 'man' or cls == 'woman' or cls == 'person':
                i1 = image_feat[i].sum().item()
                image_feat[i] = 0
                output_label.append(1)
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)

        return image_feat, output_label