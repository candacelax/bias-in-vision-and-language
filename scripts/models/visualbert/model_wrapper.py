import torch
from allennlp.common.params import Params
from torch.nn import DataParallel
from torch.nn.modules import BatchNorm2d
from tqdm import tqdm

from allennlp.nn.util import device_mapping
from visualbert.visualbert.utils.pytorch_misc import time_batch, restore_checkpoint, print_para, load_state_dict_flexible

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.DEBUG)

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

from allennlp.models import Model
#import visualbert.models.model # to register custom models for bias embedding

# modification to ModelWrapper in VisualBERT repo as this is for inference only
class InferenceModelWrapper():
    def __init__(self, args):
        self.args = args
        self.args.fp16 = args.get("fp16", False)
        self.initialize_model(args)

        self.global_step = 0
        self.called_time = 0

    def step(self, batch, output_all_encoded_layers=True):
        with torch.no_grad():
            output_dict = self.model(**batch, output_all_encoded_layers=output_all_encoded_layers)

            if output_dict['loss'] is not None:
                loss = output_dict['loss'].mean()

                output_dict['loss'] = loss

            return output_dict

    def initialize_model(self, args):
        model_params = Params({
            'type': 'VisualBERTFixedImageEmbedding',
            'special_visual_initialize': True,
            'training_head_type': 'pretraining',
            'visual_embedding_dim': 1024,
            'output_attention_weights' : False
        })
        model = Model.from_params(vocab=None, params=model_params)
        if args.fp16:
            model.half()
            print("Using FP 16, Model Halfed")
        self.model = DataParallel(model).cuda()
        self.model.eval()

    def state_dict(self):
        if isinstance(self.model, DataParallel):
            save_dict = {"model":self.model.module.state_dict()}
        else:
            save_dict = {"model":self.model.state_dict()}
        return save_dict

    def restore_checkpoint(self, serialization_dir: str, epoch_to_load: int):
        # Restore from a training dir
        return restore_checkpoint(self.model, self.optimizer, serialization_dir, epoch_to_load)

    def restore_checkpoint_pretrained(self, restore_bin: str):
        # Restore from a given model path
        state_dict = torch.load(restore_bin, map_location=device_mapping(-1))
        if isinstance(self.model, DataParallel):
            model_to_load = self.model.module
        else:
            model_to_load = self.model

        own_state = model_to_load.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                print("Skipped:" + name)
                continue
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except:
                print("Part load failed: " + name)

    def freeze_detector(self):
        if hasattr(self.model.module, "detector"):
            detector = self.model.module.detector
            for submodule in detector.backbone.modules():
                if isinstance(submodule, BatchNorm2d):
                    submodule.track_running_stats = False
                for p in submodule.parameters():
                    p.requires_grad = False
        else:
            print("No detector found.")

    @staticmethod
    def read_and_insert_args(args, confg):
        import commentjson
        from attrdict import AttrDict
        with open(confg) as f:
            config_json = commentjson.load(f)
        dict_args = vars(args)
        config_json.update(dict_args)
        args = AttrDict(config_json)
        args.model.bert_model_name = args.bert_model_name
        return args
    
