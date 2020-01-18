from visualbert.models.model_wrapper import ModelWrapper as VisualBertModel

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
                output = self.model.step(batch, eval_mode=True)
                for idx, out in enumerate(output['sequence_output']):
                    out = out.detach().cpu()
                    input_ids = batch['bert_input_ids'][idx].cpu().tolist()

                    # FIXME remove hardcode
                    # skip SOS, CLS, SEP, period tokens
                    end_idx = input_ids.index(102)
                    input_ids = input_ids[:end_idx]
                    input_ids.remove(101)
                    if 1012 in input_ids: # period token
                        input_ids.remove(1012)
                    
                    # skip SOS, CLS, END tokens
                    word_ids = set(input_ids).difference(\
                                        dataloader.dataset.general_indices)
                    contextual_idx = input_ids.index(word_ids.pop())
                    # TODO decide how to handle names OOV
                    
                    # take 0-th dim corresponding to CLS token (for words + sents)
                    cls[len(cls)] = out[0,:]
                    contextual[len(contextual)] = out[contextual_idx,:]
            return cls, contextual
        else:
            raise Exception('Other model types not yet implemented')
