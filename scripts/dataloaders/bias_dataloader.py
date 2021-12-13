from copy import deepcopy
import re
from typing import Dict, List
import torch
from torch.utils.data import DataLoader, Dataset

class BiasDataLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_gpus: int,
        category: str,
        contextual_words: List[str],
        num_workers: int=0,
        mask_token: str='[MASK]',
        pad_token: str='[PAD]'
        ):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size // num_gpus,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=getattr(dataset, 'collate_fn', None),
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
        batch = deepcopy(batch)
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