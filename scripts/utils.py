import json
from configargparse import ArgumentParser
import commentjson
from attrdict import AttrDict

def load_json(sent_file):
    ''' Load from json. We expect a certain format later, so do some post processing '''
    all_data = json.load(open(sent_file, 'r'))
    data = {}
    for k, v in all_data.items():
        captions = v["captions"]
        data[k] = captions
        v["captions"] = captions
    return all_data

def _getConfigFile():
    parser = ArgumentParser()
    parser.add_argument('--config', default='configs/visualbert_cocopretrained_coco_images.yaml',
                        help='path to model config')
    return parser.parse_args().config

def loadParams():
    config_fpath = _getConfigFile() 
    
    parser = ArgumentParser(default_config_files=[config_fpath])
    # general
    parser.add_argument('--config', type=str, help='config file path')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--val_workers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--tests', nargs='+', required=True,
                        help='paths to tests to run')
    parser.add_argument('--features', nargs='+', required=True,
                        help='paths to features (corresponding order to tests); if only one passed, will use same features for all tests')

    # dataset
    parser.add_argument('--dataset', default='coco', type=str,
                        help='name of dataset to load')
    parser.add_argument('--coco_ontology', type=str, help='only required for COCO dataset')
    # model
    parser.add_argument('--model_type', type=str, required=True, choices=['visualbert', 'vilbert'])
    parser.add_argument('--model_archive', type=str, required=True, help='path to saved model to load')
    parser.add_argument('--model_config', type=str, help='path to additional, model-specific configs')
    parser.add_argument('--bert_model', type=str, default='bert-case-uncased') # vilbert only
    parser.add_argument('--do_lower_case', action='store_true')
    parser.add_argument('--max_seq_length', type=int, default=36)
    parser.add_argument('--bert_cache', type=str, default='.pytorch_pretrained_bert')

    # bias params
    parser.add_argument('--task_cfg', type=str)
    parser.add_argument('--fpath_stopwords', type=str, default='scripts/stopwords.txt')
    parser.add_argument('--num_samples', type=int, default=100000,
                        help='num/samples for p-val permutation test')
    args = parser.parse_args()
    params = AttrDict({k:getattr(args, k) for k in vars(args)})
    
    if params.model_config: # model-specific params
        with open(params.model_config) as f:
            model_config = commentjson.load(f)
            params.update(model_config)

    return params
