import re,yaml,sys

# VilBERT
sys.path.append('vilbert_beta')
from vilbert.vilbert import VILBertForVLTasks as VILBERT

# VisualBERT
from visualbert.dataloaders.vcr import VCRLoader
from visualbert.models.model_wrapper import ModelWrapper
from visualbert.dataloaders.bias_dataset import BiasDataset as BiasDatasetVisualBERT
from vilbert.datasets.bias_dataset import BiasLoader

# one of these imports updates registrable in params.py in allennlp
from allennlp.models import Model
from visualbert.models.model_wrapper import ModelWrapper
from visualbert.models import model
from scripts.utils import load_json
from copy import deepcopy

def load_data(model_params, fp=None): # TODO rename args
    # general parameters
    data = load_json(fp)
    params_targ1 = deepcopy(model_params)
    params_targ2 = deepcopy(model_params)
    params_attr1_male_images = deepcopy(model_params)
    params_attr1_female_images = deepcopy(model_params)
    params_attr2_male_images = deepcopy(model_params)
    params_attr2_female_images = deepcopy(model_params)
    
    params_targ1.update({'data_type' : 'targ',
                         'image_prefix' : data['targ1']['image_prefix'],
                         'category' : data['targ1']['category'],
                         'examples' : data['targ1']['examples']})
    params_targ2.update({'data_type' : 'targ',
                         'image_prefix' : data['targ2']['image_prefix'],
                         'category' : data['targ2']['category'],
                         'examples' : data['targ2']['examples']})
    
    params_attr1_male_images.update({'data_type' : 'attr',
                                     'image_prefix' : data['attr1']['image_prefix'],
                                     'category' : data['attr1']['category'],
                                     'label' : 'male_images',
                                     'examples' : data['attr1']['examples']['male_images']})
    params_attr1_female_images.update({'data_type' : 'attr',
                                       'image_prefix' : data['attr1']['image_prefix'],
                                       'category' : data['attr1']['category'],
                                       'label' : 'female_images',
                                       'examples' : data['attr1']['examples']['female_images']})
    
    params_attr2_male_images.update({'data_type' : 'attr',
                                     'image_prefix' : data['attr2']['image_prefix'],
                                     'category' : data['attr2']['category'],
                                     'label' : 'male_images',
                                     'examples' : data['attr2']['examples']['male_images']})
    
    params_attr2_female_images.update({'data_type' : 'attr',
                                       'image_prefix' : data['attr2']['image_prefix'],
                                       'category' : data['attr2']['category'],
                                       'label' : 'female_images',
                                       'examples' : data['attr2']['examples']['female_images']})

    # model-specific datasets
    targ1 = create_dataloader(params_targ1)
    targ2 = create_dataloader(params_targ2)
    attr1_male_images = create_dataloader(params_attr1_male_images)
    attr1_female_images = create_dataloader(params_attr1_female_images)
    attr2_male_images = create_dataloader(params_attr2_male_images)
    attr2_female_images = create_dataloader(params_attr2_female_images)
    
    return {'targ1' : targ1,
            'targ2' : targ2,
            'attr1_male_images' : attr1_male_images,
            'attr1_female_images' : attr1_female_images,
            'attr2_male_images' : attr2_male_images,  
            'attr2_female_images' : attr2_female_images}

def create_dataloader(params):
    if params.model_type.lower() == 'visualbert':
        dataset = BiasDatasetVisualBERT(params)
        loader_params = {'batch_size': params.train_batch_size // params.num_gpus,
                         'num_gpus': params.num_gpus,
                         'num_workers': params.num_workers}
        return VCRLoader.from_dataset(dataset, **loader_params)
    elif model_params.model_type.lower() == 'vilbert':
        dataloader = BiasLoaderTrain(
            args.train_file,
            tokenizer,
            seq_len=args.max_seq_length,
            batch_size=args.train_batch_size,
            predict_feature=args.predict_feature,
            num_workers=args.num_workers,
            distributed=args.distributed)
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
