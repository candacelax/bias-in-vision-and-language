from attrdict import AttrDict
from copy import deepcopy
import re
from typing import Iterable, Dict, List, Set, Union
import torch
import torch.nn as nn
from warnings import warn

class ModelWrapper:
    def _format_output_single_stream(
        self,
        masked_t_input_ids: Union[List[int], torch.Tensor],
        mask_token_id: int,
        sequence_output: torch.Tensor,
        masked_t_sequence_output: torch.Tensor,
        masked_v_sequence_output: torch.Tensor,
        enc_full_seq: Dict[int, torch.Tensor],
        enc_contextual: Dict[int, torch.Tensor],
        enc_mask_v_full_seq: Dict[int, torch.Tensor],
        enc_mask_t_full_seq: Dict[int, torch.Tensor]
        ):
        sequence_output = sequence_output.detach().cpu()
        for idx in range(len(sequence_output)):
            if self.bidirectional:
                # take 0-th dim corresponding to CLS token (for words + sents)
                enc_full_seq[len(enc_full_seq)] = sequence_output[idx][0,:]
            else:
                # take last dim corresponding to final token
                enc_full_seq[len(enc_full_seq)] = sequence_output[idx][-1,:]

            if masked_t_input_ids is not None:
                index_of_contextual_id = (masked_t_input_ids[idx] == mask_token_id).nonzero()[0].item() # TODO add dim
                enc_contextual[len(enc_contextual)] = sequence_output[idx][index_of_contextual_id,:]
            if masked_t_sequence_output is not None:
                enc_mask_t_full_seq[len(enc_mask_t_full_seq)] = masked_t_sequence_output[idx][0,:]
            if masked_v_sequence_output is not None:
                enc_mask_v_full_seq[len(enc_mask_t_full_seq)] = masked_v_sequence_output[idx][0,:]

    def _format_output_two_stream(
        self,
        masked_t_input_ids: Union[List[int], torch.Tensor],
        mask_token_id: int,
        sequence_output_t: torch.Tensor,
        sequence_output_v: torch.Tensor,
        masked_v_sequence_output_t: torch.Tensor,
        masked_v_sequence_output_v: torch.Tensor,
        masked_t_sequence_output_t: torch.Tensor,
        masked_t_sequence_output_v: torch.Tensor,
        enc_full_seq: Dict[int, torch.Tensor],
        enc_contextual: Dict[int, torch.Tensor],
        enc_mask_v_full_seq: Dict[int, torch.Tensor],
        enc_mask_t_full_seq: Dict[int, torch.Tensor]
        ):
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

            # get index of relevant contextual ids; first index of MASK for sub-word tokenized contextual words
            index_of_target_id = (masked_t_input_ids[idx] == mask_token_id).nonzero()[0].item()
                    
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

class VisualBertWrapper(ModelWrapper):
    @staticmethod
    def add_model_args(argparser):
        parser = argparser.add_argument_group('VisualBERT Arguments')
        parser.add_argument('--bert_cache', type=str, default='.pytorch_pretrained_bert', help='specify where to download and cache')
        parser.add_argument('--coco_ontology', type=str, required=True, help='only required for processing COCO dataset')
        parser.set_defaults(bert_model_name='bert-base-uncased')
        parser.set_defaults(do_lower_case=True)
        parser.set_defaults(max_seq_length=128)
        parser.set_defaults(image_feature_cap=144)

    def __init__(self, params: AttrDict):
        from scripts.models.visualbert import VisualBERTInferenceModelWrapper
        self.model = VisualBERTInferenceModelWrapper(params)
        self.model.restore_checkpoint_pretrained(params.model_archive)
        self.bidirectional = True

    def encode(self, dataloader: Iterable):
        enc_full_seq = {} # either word or sentence (depending on input)
        enc_contextual = {} # word in context
        enc_mask_t_full_seq = {} # relevant text indices masked
        enc_mask_t_contextual = {}

        for batch_full_seq in dataloader:
            # 1. with full access to all tokens and all image regions
            output = self.model.step(batch_full_seq, output_all_encoded_layers=True)
            sequence_output = output['sequence_output'][-1].cpu().detach()

            # 2. with full access to all regions and masked language tokens
            batch_masked_t = dataloader.mask_contextual_words_in_batch(batch_full_seq, input_id_key='bert_input_ids')
            output = self.model.step(batch_masked_t, output_all_encoded_layers=True)
            masked_t_sequence_output = output['sequence_output'][-1].cpu()

            self._format_output_single_stream(
                masked_t_input_ids=batch_masked_t['bert_input_ids'].cpu().detach(),
                mask_token_id=dataloader.mask_token_id,
                sequence_output=sequence_output,
                masked_t_sequence_output=masked_t_sequence_output,
                masked_v_sequence_output=None,
                enc_full_seq=enc_full_seq,
                enc_contextual=enc_contextual,
                enc_mask_t_full_seq=enc_mask_t_full_seq,
                enc_mask_v_full_seq=False,
                )

        enc = {'full_seq' : enc_full_seq, 'contextual' : enc_contextual}
        enc_mask_t = {'full_seq' : enc_mask_t_full_seq, 'contextual' : enc_mask_t_contextual}
        # no visual masking for VisualBERT so set to empty dicts
        enc_mask_v = {'full_seq' : {}, 'contextual' : {}}
        return enc, enc_mask_t, enc_mask_v

    def predict_words(self, dataloader: Iterable, k: int=3):
        for batch in dataloader:
            batch_size, lang_dim = batch["bert_input_ids"].shape
            output = self.model.step(batch, eval_mode=True,
                                     output_all_encoded_layers=True)
            logits = output['logits']
            _,predicted = logits[:,:lang_dim].topk(k=k, dim=-1)
            for orig, pred, label in zip(batch["bert_input_ids"],
                                         predicted,
                                         batch["masked_lm_labels"]):
                gt_ids = label[label!=-1]
                gt_tokens = dataloader.tokenizer.convert_ids_to_tokens(gt_ids.tolist())
                pred_ids = pred[(label!=-1).nonzero()].flatten()
                pred_tokens = dataloader.tokenizer.convert_ids_to_tokens(pred_ids.tolist())
    

class ViLBERTWrapper(ModelWrapper):
    @staticmethod
    def add_model_args(argparser):
        parser = argparser.add_argument_group('ViLBERT Arguments')
        parser.add_argument('--model_config', type=str, help='path to additional, model-specific configs')
        parser.add_argument('--path_to_obj_list', type=str, required=True, help='path to list of objects by idx; needed for image region masking')
        parser.add_argument('--dataset_type', type=str, required=True, choices=['concap', 'google'])
        parser.set_defaults(bert_model_name='bert-base-uncased')
        parser.set_defaults(do_lower_case=True)

    def __init__(self, params: AttrDict):
        from scripts.models.vilbert import ViLBERTModel, ViLBERTConfig

        config = ViLBERTConfig.from_json_file(params.model_config)
        self.model = ViLBERTModel.from_pretrained(
            pretrained_model_name_or_path=params.model_archive,
            config=config
            )
        self.model.cuda()
        self.model.eval()
        self.bidirectional = True
        
        self.BATCH_KEYS = [
            'input_ids', 'input_mask', 'segment_ids', 'lm_label_ids', 'image_feat', 'image_loc', \
            'image_label', 'image_mask', 'image_ids', 'coattention_mask',\
            'masked_image_feat', 'masked_image_label'
            ]
        
    def encode(self, dataloader: Iterable):
        enc_full_seq = {} # either word or sentence (depending on input)
        enc_contextual = {} # word in context
        enc_mask_t_full_seq = {} # relevant text indices masked
        enc_mask_t_contextual = {}
        enc_mask_v_full_seq = {} # relevant image regions masked
        enc_mask_v_contextual = {}
        
        for batch in dataloader:
            batch = {key:tensor.cuda(non_blocking=True) for key, tensor in zip(self.BATCH_KEYS, batch)}

            # 1. with full access to all tokens and all image regions
            output = self.model(
                batch['input_ids'],
                batch['image_feat'],
                batch['image_loc'],
                batch['segment_ids'],
                return_sequence_output=True,
                output_all_attention_masks=True
                )
            attention_mask, sequence_output_t, sequence_output_v = output[-3:]

            # 2. with full access to all tokens and masked image regions)
            #masked_v_batch = dataloader.mask_image_features_in_batch(deepcopy(batch), input_id_key='input_ids')
            masked_v_output = self.model(
                batch['input_ids'],
                batch['masked_image_feat'],
                batch['image_loc'],
                batch['segment_ids'],
                return_sequence_output=True
                )
            masked_v_sequence_output_t, masked_v_sequence_output_v = masked_v_output[-2:]

            # 3. with full access to all regions and masked language tokens
            masked_t_batch = dataloader.mask_contextual_words_in_batch(deepcopy(batch), input_id_key='input_ids')
            masked_t_output = self.model(
                masked_t_batch['input_ids'],
                masked_t_batch['image_feat'],
                masked_t_batch['image_loc'],
                masked_t_batch['segment_ids'],
                return_sequence_output=True
                )
            masked_t_sequence_output_t, masked_t_sequence_output_v = masked_t_output[-2:]
            self._format_output_two_stream(
                masked_t_batch['input_ids'],
                dataloader.mask_token_id,
                sequence_output_t,
                sequence_output_v,
                masked_v_sequence_output_t,
                masked_v_sequence_output_v,
                masked_t_sequence_output_t,
                masked_t_sequence_output_v,
                enc_full_seq,
                enc_contextual,
                enc_mask_v_full_seq,
                enc_mask_t_full_seq
                )

        enc = {'full_seq' : enc_full_seq, 'contextual' : enc_contextual}
        enc_mask_v = {'full_seq' : enc_mask_v_full_seq, 'contextual' : enc_mask_v_contextual}
        enc_mask_t = {'full_seq' : enc_mask_t_full_seq, 'contextual' : enc_mask_t_contextual}

        return enc, enc_mask_t, enc_mask_v

class LXMERTWrapper(ModelWrapper): # HuggingFace implementation
    @staticmethod
    def add_model_args(argparser):
        parser = argparser.add_argument_group('Lxmert Arguments')
        parser.add_argument('--bert_model_name', type=str, default='unc-nlp/lxmert-base-uncased')
        parser.add_argument('--path_to_obj_list', type=str, required=True, help='path to list of objects by idx; needed for image region masking')

    def __init__(self, params: AttrDict):
        from scripts.models.lxmert import LxmertForPreTrainingBias
        self.model = LxmertForPreTrainingBias.from_pretrained(params.bert_model_name, return_dict=True)
        self.model.cuda()
        self.model.eval()
        self.bidirectional = True

    def encode(self, dataloader: Iterable):
        enc_full_seq = {} # either word or sentence (depending on input)
        enc_contextual = {} # word in context
        enc_mask_t_full_seq = {} # relevant text indices masked
        enc_mask_t_contextual = {}
        enc_mask_v_full_seq = {} # relevant image regions masked
        enc_mask_v_contextual = {}

        for batch_full_access in dataloader:
            # 1. with full access to all tokens and all image regions
            #batch_full_access = dataloader.format_batch(deepcopy(batch))
            obj_indices = batch_full_access.pop('obj_indices')
            output = self.model(
                **batch_full_access,
                output_attentions=True,
                output_hidden_states=True,
                return_outputs=True,
                return_sequence_output=True
                )
            sequence_output_t = output.lang_output.detach().cpu()
            sequence_output_v = output.visual_output.detach().cpu()

            # 2. with full access to all tokens and masked image regions
            batch_masked_image_regions = dataloader.mask_image_regions(deepcopy(batch_full_access), obj_indices)
            masked_image_output = self.model(
                **batch_masked_image_regions,
                output_attentions=True,
                output_hidden_states=True,
                return_outputs=True,
                return_sequence_output=True
            )
            masked_v_sequence_output_t = masked_image_output.lang_output.detach().cpu()
            masked_v_sequence_output_v = masked_image_output.visual_output.detach().cpu()
            
            # 3. with full access to all regions and masked language tokens
            #batch_masked_tokens = dataloader.format_batch(deepcopy(batch), mask_contextual_words=True)
            batch_masked_tokens = dataloader.mask_contextual_words_in_batch(deepcopy(batch_full_access), 'input_ids')
            masked_t_input_ids = batch_masked_tokens['input_ids'] # we'll use this later to find the relevant contextual ids
            masked_token_output = self.model(
                **batch_masked_tokens,
                output_attentions=True,
                output_hidden_states=True,
                return_outputs=True,
                return_sequence_output=True
            )
            masked_t_sequence_output_t = masked_token_output.lang_output.detach().cpu()
            masked_t_sequence_output_v = masked_token_output.visual_output.detach().cpu()

            self._format_output_two_stream(
                masked_t_input_ids,
                dataloader.mask_token_id,
                sequence_output_t, sequence_output_v,
                masked_v_sequence_output_t, masked_v_sequence_output_v,
                masked_t_sequence_output_t, masked_t_sequence_output_v,
                enc_full_seq, enc_contextual,
                enc_mask_v_full_seq, enc_mask_t_full_seq
                )
            
        enc = {'full_seq' : enc_full_seq, 'contextual' : enc_contextual}
        enc_mask_v = {'full_seq' : enc_mask_v_full_seq, 'contextual' : enc_mask_v_contextual}
        enc_mask_t = {'full_seq' : enc_mask_t_full_seq, 'contextual' : enc_mask_t_contextual}
        return enc, enc_mask_t, enc_mask_v

class VLBERTWrapper(ModelWrapper):
    @staticmethod
    def add_model_args(argparser):
        parser = argparser.add_argument_group('VL-Bert Arguments')
        parser.add_argument('--bert_cache', type=str, default='.pytorch_pretrained_bert', help='specify where to download and cache')
        parser.add_argument('--model_config_path', type=str, help='path to additional, model-specific configs') # TODO remove?
        parser.add_argument('--path_to_obj_list', type=str, required=True, help='path to list of objects by idx; needed for image region masking')

    def __init__(self, params: AttrDict):
        from scripts.models.vlbert import vlbert_model_config, update_vlbert_config, ResNetVLBERTForPretraining

        update_vlbert_config(params.model_config_path)
        self.model = ResNetVLBERTForPretraining(vlbert_model_config, params.model_archive) #vlbert_model_config)
        #checkpoint = torch.load(params.model_archive, map_location=lambda storage, loc: storage)['state_dict']
        #checkpoint.update({
        #    'object_mask_visual_embedding.weight' : torch.zeros_like(self.model.object_mask_visual_embedding.weight),
        #    #'object_mask_visual_embedding.bias' : torch.zeros_like(self.model.object_mask_visual_embedding.bias)
        #    #'image_feature_extractor.obj_downsample.1.weight' : self.model.image_feature_extractor.obj_downsample.1.weight
        #    }
        #})
        #for k,v in self.model.state_dict().items():
        #    if k not in checkpoint:
        #        k = 'module.' + k
        #        if k not in checkpoint:
        #            warn(f'Key {k} is missing from pretrained model')
        #    setattr(self.model, k, v)
        self.model.cuda()
        self.model.eval()
        self.bidirectional = True

    def encode(self, dataloader: Iterable):
        self.model.eval()
        enc_full_seq = {} # either word or sentence (depending on input)
        enc_contextual = {} # word in context
        enc_mask_t_full_seq = {} # relevant text indices masked
        enc_mask_t_contextual = {}
        enc_mask_v_full_seq = {} # relevant image regions masked
        enc_mask_v_contextual = {}
        
        text_index = dataloader.dataset.data_names.index('text')
        boxes_index = dataloader.dataset.data_names.index('boxes')
        obj_labels_index = dataloader.dataset.data_names.index('object_labels')
        for batch in dataloader:
            # 1. with full access to all tokens and all image regions
            batch = [v.cuda() if isinstance(v, torch.Tensor) else v for v in batch]
            output = self.model(*batch[:-1]) # pass everything as input except object labels
            sequence_output = output['sequence_output'].cpu().detach()
            input_ids = batch[text_index].detach().cpu().clone()

            # 2. with full access to all tokens and masked image regions
            masked_v_batch = deepcopy(batch)
            boxes = dataloader.mask_input_features(masked_v_batch[boxes_index], masked_v_batch[obj_labels_index])
            masked_v_batch[boxes_index] = boxes
            output = self.model(*masked_v_batch[:-1]) # pass everything as input except object labels
            masked_v_sequence_output = output['sequence_output'].cpu().detach()

            # 3. with full access to all regions and masked language tokens
            masked_t_batch = deepcopy(batch)
            masked_input_ids = dataloader.mask_input_ids(masked_t_batch[text_index])
            masked_t_batch[text_index] = masked_input_ids
            output = self.model(*masked_t_batch[:-1]) # pass everything as input except object labels
            masked_t_sequence_output = output['sequence_output'].cpu().detach()

            self._format_output_single_stream(
                masked_input_ids.detach().cpu(),
                mask_token_id=dataloader.mask_token_id,
                sequence_output=sequence_output,
                masked_t_sequence_output=masked_t_sequence_output,
                masked_v_sequence_output=masked_v_sequence_output,
                enc_full_seq=enc_full_seq,
                enc_contextual=enc_contextual,
                enc_mask_t_full_seq=enc_mask_t_full_seq,
                enc_mask_v_full_seq=enc_mask_v_full_seq

            )
        enc = {'full_seq' : enc_full_seq, 'contextual' : enc_contextual}
        enc_mask_v = {'full_seq' : enc_mask_v_full_seq, 'contextual' : enc_mask_v_contextual}
        enc_mask_t = {'full_seq' : enc_mask_t_full_seq, 'contextual' : enc_mask_t_contextual}
        return enc, enc_mask_t, enc_mask_v


# define types
TYPE2WRAPPER = {
    'visualbert' : VisualBertWrapper,
    'vilbert' : ViLBERTWrapper,
    'lxmert' : LXMERTWrapper,
    'vlbert' : VLBERTWrapper
}