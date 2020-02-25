import torch
from visualbert.models.model_wrapper import ModelWrapper as VisualBertModel
from vilbert.task_utils import EvaluatingModel as ViLBERTEval

class EncoderWrapper:
    def __init__(self, model):
        self.model = model

    def encode(self, dataloader):
        if isinstance(self.model, VisualBertModel):
            # --- VisualBERT
            self.model.eval()

            cls = {} # either word or sentence (depending on input)
            contextual = {} # word in context
            for batch in dataloader:
                batch['output_all_encoded_layers'] = True
                output = self.model.step(batch, eval_mode=True)

                sequence_output = output['sequence_output'][-1]
                for idx, out in enumerate(sequence_output):
                    out = out.detach().cpu()
                    input_ids = batch['bert_input_ids'][idx].cpu().tolist()

                    word_ids = [i for i in input_ids
                                if i not in dataloader.dataset.stopword_indices]
                    contextual_idx = input_ids.index(word_ids[0])
                    
                    # take 0-th dim corresponding to CLS token (for words + sents)
                    cls[len(cls)] = out[0,:]
                    contextual[len(contextual)] = out[contextual_idx,:]
            return cls, contextual
        else:
            self.model.eval()

            cls = {} # either word or sentence (depending on input)
            contextual = {} # word in context
            for batch in dataloader:
                batch = tuple(t.cuda(non_blocking=True) for t in batch)
                features, spatials, image_mask, question, input_mask, \
                    segment_ids, co_attention_mask, question_id = batch
                batch_size = features.size(0)

                with torch.no_grad():
                    vil_prediction, vil_logit, vil_binary_prediction, vision_prediction,\
                        vision_logit, linguisic_prediction, linguisic_logit, \
                        sequence_out \
                        = self.model(question, features, spatials, segment_ids, input_mask,
                                     image_mask, co_attention_mask)

                    for idx, out in enumerate(sequence_out):
                        out = out.detach().cpu()
                        input_ids = question[idx].cpu().tolist()
                        word_ids = [i for i in input_ids
                                    if i not in dataloader.dataset.stopword_indices]
                        contextual_idx = input_ids.index(word_ids[0])
                    
                        # take 0-th dim corresponding to CLS token (for words + sents)
                        cls[len(cls)] = out[0,:]
                        contextual[len(contextual)] = out[contextual_idx,:]
            return cls, contextual
