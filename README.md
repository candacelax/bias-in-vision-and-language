# Measuring Social Biases in Grounded Visual and Language Embeddings
This is the repo for our paper [Measuring Social Biases in Grounded Vision and Language Embeddings](https://arxiv.org/abs/2002.08911). We implement a version of WEAT/SEAT for visually grounded word embeddings. This is code borrowed and modified from [this repo](https://github.com/W4ngatang/sent-bias). Authors: [Candace Ross](candaceross.io), [Boris Katz](https://www.csail.mit.edu/person/boris-katz), [Andrei Barbu](0xab.com)

## Installation
Create the conda environment.
```bash
git clone git@github.com:candacelax/vilbert_beta.git # forked
git clone git@github.com:candacelax/visualbert.git # forked

git clone git@github.com:candacelax/bias-in-vision-and-language.git
cd bias-grounded-bert
ln -s PATH_TO_VISUALBERT visualbert
ln -s PATH_TO_VILBERT vilbert_beta

conda env create -f conda_env.yml
python -m spacy download en

# add VisualBERT and ViLBERT to Python paths (consider doing in .bashrc)
export PYTHONPATH="$PYTHONPATH:XX/bias-grounded-bert/visualbert"
export PYTHONPATH="$PYTHONPATH:XX/bias-grounded-bert/vilbert_beta"
export PYTORCH_PRETRAINED_BERT_CACHE="XX/bias-grounded-bert/.pytorch_pretrained_bert"
```


Download the pretrained models for [ViLBERT](https://drive.google.com/drive/folders/1Re0L75uazH3Qrep_aRgtaVelDEz4HV9c) and [VisualBERT](https://drive.google.com/file/d/1QvivVfRsRF518OQSQNaN7aFk6eQ43vP_/view). Save each in the model's respective pretrained-models directory.

## Download data
The paths for images from Google Image Search are contained in `data/google-images`. To download all at once, run
```bash
   ./scripts/download_data.sh data/google-images
```

## Usage
To run GWEAT/GSEAT tests, run:
`./main.py --config CONFIG_FILEPATH`

The config files are contained in `configs` are include tests of Conceptual Captions and Google Image data on pretrained ViLBERT and tests of COCO and Google Image data on pretrained VisualBERT.

## Feature Extraction
If you want to run over custom images, you'll need to compute features. We use the same approach from each respective paper.

### VisualBERT
VisualBERT uses [Detectron](https://github.com/facebookresearch/Detectron) to get features from faster-rcnn.
```bash
   git clone git@github.com:facebookresearch/Detectron.git
   mkdir Detectron/pretrained-models
   mv Detectron visualbert/utils

   # download pretrained model
   wget -o visualbert/utils/Detectron/pretrained-models/detectron_35861858.pkl https://dl.fbaipublicfiles.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl

   # run feature extract example
   python visualbert/utils/get_image_features/extract_image_features_nlvr.py \
   	  --im_or_folder data/XX/BIAS-TEST-IMAGES \
	  --one_giant_file visualbert/image-features/XX/BIAS_TEST_NAME.th \
	  --output_dir temp \
	  --cfg visualbert/utils/Detectron/configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
	  --wts visualbert/utils/Detectron/pretrained-models/detectron_35861858.pkl \
	  --existing visualbert/image-features/XX/BIAS_TEST_NAME.th # ONLY IF UPDATING PREVIOUS RUNS
   
```

### ViLBERT
Clone [bottom-up-attention](https://github.com/jiasenlu/bottom-up-attention) (be sure to use their forked version) and create Docker image of Caffe.
```bash
	BASE_DIR=XX/SET_TO_LOCATION_OF_DOCKER
	DATA_DIR=XX/vilbert_beta/data/conceptual-captions
	OUTPUT_FILE=XX/vilbert_beta_features/conceptual_val_resnet101_faster_rcnn_genome.tsv
	SPLIT=validation

	# create docker image
 	docker build -f bottom-up-attention/caffe/docker/standalone/gpu/Dockerfile -t caffe_image_features .

   	# run container
	docker container run -t -v $BASE_DIR/features:$BASE_DIR/features \
	     --gpus all caffe_image_features \
	     python2.7 $BASE_DIR/tools/generate_tsv.py --cfg $BASE_DIR/experiments/cfgs/faster_rcnn_end2end_resnet.yml \
	     	       --def $BASE_DIR/models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt \
		       --net $BASE_DIR/models/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel \
		       --total_group 1 --group_id 0 --split $SPLIT \
		       --gpu 0,1,2,3,4,5,6,7 --out $OUTPUT_FILE --data_dir $DATA_DIR

        # convert features to LMDB
	sudo chown -R $USER:$USER $(dirname $OUTPUT_FILE)
	python scripts/convert_general_lmdb.py $OUTPUT_FILE
```