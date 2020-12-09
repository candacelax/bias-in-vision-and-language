from attrdict import AttrDict
from typing import Dict, List
from . import bias_dataloader

def create_dataloader(
    params: AttrDict,
    category: str,
    captions: Dict[str, str],
    images: Dict[str, List[int]],
    contextual_words: List[str],
    image_features=None,
    ):
    assert params.model_type in bias_dataloader.MODEL2LOADER, f'Unknown model type: {params.model_type}'
    loader_class = getattr(bias_dataloader, bias_dataloader.MODEL2LOADER[params.model_type])
    return loader_class(
        params=params,
        category=category,
        captions=captions,
        images=images,
        contextual_words=contextual_words,
        image_features=image_features
        )