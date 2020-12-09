
from configargparse import ArgumentParser, YAMLConfigFileParser
from attrdict import AttrDict
from time import localtime
import logging as log
from os import makedirs, path
import torch

def load_query_params():
    parser = ArgumentParser(config_file_parser_class=YAMLConfigFileParser)
    # general
    parser.add_argument('-c', '--config', is_config_file=True, required=True, type=str,
                        help='config file path')
    parser.add_argument('--out_dir', type=str, default='results', help='path to save results')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--val_workers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64)

    # bias tests and image/text data
    parser.add_argument('--coco_ontology', type=str, help='only required for COCO dataset')
    parser.add_argument('--path_to_obj_list', type=str,
                        help='path to list of objects by idx; only needed for ViLBERT')
    # model
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['lxmert', 'visualbert', 'vilbert', 'vlbert'])
    parser.add_argument('--model_archive', type=str, required=True, help='path to saved model to load')
    parser.add_argument('--model_config', type=str, help='path to additional, model-specific configs')
    parser.add_argument('--bert_model', type=str, default='bert-case-uncased')
    parser.add_argument('--do_lower_case', action='store_true')
    parser.add_argument('--max_seq_length', type=int, default=36)
    parser.add_argument('--bert_cache', type=str, default='.pytorch_pretrained_bert')

    args = parser.parse_args()

    # additional arguments, check dirs
    args.num_gpus = torch.cuda.device_count()
    args.is_train = False
    makedirs(args.out_dir, exist_ok=True)
    makedirs(path.join(args.out_dir, args.model_type), exist_ok=True)

    
    params = AttrDict({k:getattr(args, k) for k in vars(args)})
    return params


def setup_logging_results(params: AttrDict):
    t = localtime()
    timestamp = f'{t.tm_mon}-{t.tm_mday}-{t.tm_year}_' +\
                f'{t.tm_hour:02d}:{t.tm_min:02d}:{t.tm_sec:02d}'
    model_log_dir = path.join(params.log_dir, params.model_type)
    log_fpath = path.join(params.log_dir, params.model_type, timestamp)
    save_dir = path.join(params.out_dir, params.model_type, timestamp)

    makedirs(model_log_dir, exist_ok=True)
    makedirs(save_dir, exist_ok=True)
    
    log.getLogger().addHandler(log.FileHandler(log_fpath))
    log.info(f'Params: {params}')

    return log, save_dir, timestamp
    
