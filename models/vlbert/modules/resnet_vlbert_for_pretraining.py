import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..external.pytorch_pretrained_bert import BertTokenizer
from ..common.module import Module
from ..common.fast_rcnn import FastRCNN
from ..common.visual_linguistic_bert import VisualLinguisticBertForPretraining
from ..common.utils.misc import soft_cross_entropy

class ResNetVLBERTForPretraining(Module):
    def __init__(self, config, pretrained_model_path):
        super(ResNetVLBERTForPretraining, self).__init__(config)
        self.image_feature_extractor = FastRCNN(config,
                                                average_pool=True,
                                                final_dim=config.NETWORK.IMAGE_FINAL_DIM,
                                                enable_cnn_reg_loss=False)
        self.object_linguistic_embeddings = nn.Embedding(1, config.NETWORK.VLBERT.hidden_size)
        self.object_mask_visual_embedding = nn.Embedding(1, 2048) # image features precomputed
        self.image_feature_bn_eval = True # backbone is frozen
        self.tokenizer = BertTokenizer.from_pretrained(config.NETWORK.BERT_MODEL_NAME)
        self.vlbert = VisualLinguisticBertForPretraining(
            config.NETWORK.VLBERT,
            language_pretrained_model_path=None,
            with_rel_head=False,
            with_mlm_head=False,
            with_mvrc_head=False,
        )
        pretrained_state_dict = torch.load(pretrained_model_path, map_location=lambda storage, loc: storage)['state_dict']
        # remove module from key name and pop mlm head
        pretrained_state_dict = {}
        for k,v in torch.load(pretrained_model_path, map_location=lambda storage, loc: storage)['state_dict'].items():
            k = re.sub('module.', '', k)
            if re.match('vlbert.mlm_*|vlbert.mvrc_head.*', k):
                continue
            elif re.match('object_mask_word.*|aux_text_visual_embedding.*', k):
                continue
            pretrained_state_dict[k] = v
        self.load_state_dict(pretrained_state_dict)

    def _collect_obj_reps(self, span_tags, object_reps):
        """
        Collect span-level object representations
        :param span_tags: [batch_size, ..leading_dims.., L]
        :param object_reps: [batch_size, max_num_objs_per_batch, obj_dim]
        :return:
        """

        span_tags_fixed = torch.clamp(span_tags, min=0)  # In case there were masked values here
        row_id = span_tags_fixed.new_zeros(span_tags_fixed.shape)
        row_id_broadcaster = torch.arange(0, row_id.shape[0], step=1, device=row_id.device)[:, None]

        # Add extra diminsions to the row broadcaster so it matches row_id
        leading_dims = len(span_tags.shape) - 2
        for i in range(leading_dims):
            row_id_broadcaster = row_id_broadcaster[..., None]
        row_id += row_id_broadcaster
        return object_reps[row_id.view(-1), span_tags_fixed.view(-1)].view(*span_tags_fixed.shape, -1)

    def forward(self,
                image,
                boxes,
                im_info,
                text,
                relationship_label,
                mlm_labels,
                mvrc_ops,
                mvrc_labels):
        ###########################################

        # visual feature extraction
        images = image
        box_mask = (boxes[:, :, 0] > -1.5)
        origin_len = boxes.shape[1]
        max_len = int(box_mask.sum(1).max().item())
        box_mask = box_mask[:, :max_len]
        boxes = boxes[:, :max_len]
        mvrc_ops = mvrc_ops[:, :max_len]
        mvrc_labels = mvrc_labels[:, :max_len]

        box_features = boxes[:, :, 4:]
        #box_features[mvrc_ops == 1] = self.object_mask_visual_embedding.weight[0]
        boxes[:, :, 4:] = box_features

        obj_reps = self.image_feature_extractor(images=images,
                                                boxes=boxes,
                                                box_mask=box_mask,
                                                im_info=im_info,
                                                classes=None,
                                                segms=None,
                                                mvrc_ops=mvrc_ops,
                                                mask_visual_embed=None)

        ############################################

        # prepare text
        text_input_ids = text
        text_tags = text.new_zeros(text.shape)
        text_token_type_ids = text.new_zeros(text.shape)
        text_mask = (text_input_ids > 0)
        text_visual_embeddings = self._collect_obj_reps(text_tags, obj_reps['obj_reps'])

        object_linguistic_embeddings = self.object_linguistic_embeddings(
            boxes.new_zeros((boxes.shape[0], boxes.shape[1])).long()
        )
        object_vl_embeddings = torch.cat((obj_reps['obj_reps'], object_linguistic_embeddings), -1)

        ###########################################

        # Visual Linguistic BERT
        relationship_logits, mlm_logits, mvrc_logits, sequence_output = self.vlbert(
            text_input_ids,
            text_token_type_ids,
            text_visual_embeddings,
            text_mask,
            object_vl_embeddings,
            box_mask
            )
        ###########################################
        outputs = {
            'relationship_logits': relationship_logits,
            'relationship_label': relationship_label,
            'sequence_output' : sequence_output
        }
        return outputs

