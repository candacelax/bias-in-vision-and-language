import re,yaml,sys

# VilBERT
sys.path.append('vilbert_beta')
from vilbert.vilbert import VILBertForVLTasks as VILBERT
from vilbert.datasets.bias_dataset import BiasLoader as BiasLoaderViLBERT
from pytorch_pretrained_bert.tokenization import BertTokenizer

# VisualBERT
from visualbert.dataloaders.vcr import VCRLoader
from visualbert.models.model_wrapper import ModelWrapper
from visualbert.dataloaders.bias_dataset import BiasDataset as BiasDatasetVisualBERT

# one of these imports updates registrable in params.py in allennlp
from allennlp.models import Model
from visualbert.models.model_wrapper import ModelWrapper
from visualbert.models import model
from scripts.utils import load_json
from copy import deepcopy

def load_data(model_params, fp=None): # TODO rename args
    # general parameters
    data = load_json(fp)
    params_targ_X = deepcopy(model_params)
    params_targ_Y = deepcopy(model_params)
    params_attr_A_X = deepcopy(model_params)
    params_attr_A_Y = deepcopy(model_params)
    params_attr_B_X = deepcopy(model_params)
    params_attr_B_Y = deepcopy(model_params)
    
    params_targ_X.update({'data_type' : 'target',
                          'category' : data['targ1']['category'],
                          'captions' : data['targ1']['captions'],
                          'images' : data['targ1']['images']})
    params_targ_Y.update({'data_type' : 'target',
                          'category' : data['targ2']['category'],
                          'captions' : data['targ2']['captions'],
                          'images' : data['targ2']['images']})

    category_X, category_Y = params_targ_X['category'], params_targ_Y['category']
    params_attr_A_X.update({'data_type' : 'attr',
                            'category' : data['attr1']['category'],
                            'captions' : data['attr1']['captions'],
                            'images' : data['attr1'][f'{category_X}_Images']})
    params_attr_A_Y.update({'data_type' : 'attr',
                            'category' : data['attr1']['category'],
                            'captions' : data['attr1']['captions'],
                            'images' : data['attr1'][f'{category_Y}_Images']})
    params_attr_B_X.update({'data_type' : 'attr',
                            'category' : data['attr2']['category'],
                            'captions' : data['attr2']['captions'],
                            'images' : data['attr2'][f'{category_X}_Images']})
    params_attr_B_Y.update({'data_type' : 'attr',
                            'category' : data['attr2']['category'],
                            'captions' : data['attr2']['captions'],
                            'images' : data['attr2'][f'{category_Y}_Images']})

    # model-specific datasets
    dataloader_targ_X = create_dataloader(params_targ_X)
    dataloader_targ_Y = create_dataloader(params_targ_Y)
    dataloader_A_X = create_dataloader(params_attr_A_X)
    dataloader_A_Y = create_dataloader(params_attr_A_Y)
    dataloader_B_X = create_dataloader(params_attr_B_X)
    dataloader_B_Y = create_dataloader(params_attr_B_Y)
    
    return {'targ_X' : dataloader_targ_X,
            'targ_Y' : dataloader_targ_Y,
            'attr_A_X' : dataloader_A_X,
            'attr_A_Y' : dataloader_A_Y,
            'attr_B_X' : dataloader_B_X,
            'attr_B_Y' : dataloader_B_Y}

def create_dataloader(params):
    if params.model_type.lower() == 'visualbert':
        dataset = BiasDatasetVisualBERT(params)
        loader_params = {'batch_size': params.batch_size // params.num_gpus,
                         'num_gpus': params.num_gpus,
                         'num_workers': params.num_workers}
        return VCRLoader.from_dataset(dataset, **loader_params)
    
    elif params.model_type.lower() == 'vilbert':
        tokenizer = BertTokenizer.from_pretrained(
            params.bert_model,
            do_lower_case=params.do_lower_case
        )
        dataloader = BiasLoaderViLBERT(params['examples'],
                                       tokenizer,
                                       seq_len=params.max_seq_length,
                                       batch_size=params.batch_size,
                                       predict_feature=params.get('predict_feature', False),
                                       num_workers=params.num_workers,
                                       distributed=params.get('distributed', False))
        for batch in dataloader:
            print('batch', batch)
            exit()
    else:
        raise Exception(f'{params.model_type} is an unsupported model type')
        
        
def load_model(params):
    if re.match('vilbert', params.model_type, re.IGNORECASE):
        # from vilbert.task_utils import LoadDatasetEval        
        # with open('vilbert_beta/vlbert_tasks.yml', 'r') as f:
        #     task_cfg = yaml.load(f, Loader=yaml.FullLoader)
        #     print('t', task_cfg)
        #     exit()
            
        # task_batch_size, task_num_iters, task_ids, task_datasets_val, task_dataloader_val \
        #     = LoadDatasetEval(args, task_cfg, args.tasks.split('-'))
        from vilbert.vilbert import BertForMultiModalPreTraining, BertConfig
        if params.use_concap:
            config = BertConfig.from_json_file(params.model_config)
            model = BertForMultiModalPreTraining.from_pretrained(params.model_archive,
                                                                 config)
            model = model.cuda()
            
        #model = VILBERT.from_pretrained(pretrained_model_name_or_path=params.model_archive,
        #config=params.model_config)
    elif re.match('visualbert', params.model_type, re.IGNORECASE):
        model = ModelWrapper(params, params.train_set_size)
        model.restore_checkpoint_pretrained(params.model_path)
    return model
