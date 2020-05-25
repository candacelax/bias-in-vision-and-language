from configargparse import ArgumentParser, YAMLConfigFileParser
import commentjson
from attrdict import AttrDict
from time import localtime
import logging as log
import os
import re
import scripts # TODO rename

def load_eval_params():
    parser = ArgumentParser(config_file_parser_class=YAMLConfigFileParser)
    # general
    parser.add_argument('-c', '--config', is_config_file=True, required=True, type=str,
                        help='config file path')
    parser.add_argument('--out_dir', type=str, default='results', help='path to save results')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--val_workers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64)

    # bias tests and image/text data
    parser.add_argument('--test2features_path', type=str, required=True,
                        help='path to JSON file of features, stored by model and test')
    parser.add_argument('--tests', nargs='+', required=True, help='paths to tests to run')
    parser.add_argument('--coco_ontology', type=str, help='only required for COCO dataset')
    parser.add_argument('--obj_list', type=str,
                        help='path to list of objects by idx; only needed for ViLBERT')
    # model
    parser.add_argument('--model_type', type=str, required=True, choices=['visualbert', 'vilbert'])
    parser.add_argument('--model_archive', type=str, required=True, help='path to saved model to load')
    parser.add_argument('--model_config', type=str, help='path to additional, model-specific configs')
    parser.add_argument('--bert_model', type=str, default='bert-case-uncased')
    parser.add_argument('--do_lower_case', action='store_true')
    parser.add_argument('--max_seq_length', type=int, default=36)
    parser.add_argument('--bert_cache', type=str, default='.pytorch_pretrained_bert')

    parser.add_argument('--num_samples', type=int, default=100000,
                        help='num/samples for p-val permutation test')
    args = parser.parse_args()
    params = AttrDict({k:getattr(args, k) for k in vars(args)})
    
    if params.model_config: # model-specific params
        with open(params.model_config) as f:
            model_config = commentjson.load(f)
            params.update(model_config)

    if params.model_type == 'visualbert':
        params.is_train = False

    return params

def setup_log(params: AttrDict):
    t = localtime()
    timestamp = f'{t.tm_mon}-{t.tm_mday}-{t.tm_year}_' +\
                f'{t.tm_hour:02d}:{t.tm_min:02d}:{t.tm_sec:02d}'
    log_fpath = os.path.join(params.log_dir, timestamp)
    log.getLogger().addHandler(log.FileHandler(log_fpath))
    log.info(f'Params: {params}')
    os.mkdir(f'results/{timestamp}')
    return log, timestamp

def load_model(params: AttrDict):
    # load model
    if re.match('vilbert', params.model_type, re.IGNORECASE):
        config = scripts.ViLBERTBertConfig.from_json_file(params.model_config)
        model = scripts.ViLBERTPretraining.from_pretrained(\
                    pretrained_model_name_or_path=params.model_archive, config=config).cuda()

    elif re.match('visualbert', params.model_type, re.IGNORECASE):
        params.train_set_size = 20000 # not important, only used for optim and we're not doing this step
        # TODO would be nice to entirely skip using wrapper and need to init optim
        model = scripts.VisualBERTModelWrapper(params, params.train_set_size)
        model.restore_checkpoint_pretrained(params.model_archive)

    return model
