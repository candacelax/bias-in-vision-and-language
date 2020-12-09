# Measuring Social Biases in Grounded Visual and Language Embeddings
This is the repo for our paper [Measuring Social Biases in Grounded Vision and Language Embeddings](https://arxiv.org/abs/2002.08911). We implement a version of WEAT/SEAT for visually grounded word embeddings. This is code borrowed and modified from [this repo](https://github.com/W4ngatang/sent-bias). Authors: [Candace Ross](candaceross.io), [Boris Katz](https://www.csail.mit.edu/person/boris-katz), [Andrei Barbu](0xab.com)

## Installation
Create the conda environment.
```bash
git clone git@github.com:candacelax/bias-in-vision-and-language.git
cd bias-grounded-bert

conda env create -f conda_env.yml
conda activate visual-bias
python -m spacy download en

export PYTORCH_PRETRAINED_BERT_CACHE="XX/bias-grounded-bert/.pytorch_pretrained_bert"
```

## Usage
After downloading data, pretrained models and image features (described below), to run GWEAT/GSEAT tests, run:
`./main.py --config CONFIG_FILEPATH`

The config files are contained in `configs` are include tests of Conceptual Captions and Google Image data on pretrained ViLBERT and tests of COCO and Google Image data on pretrained VisualBERT.


## Download data
The paths for images from Google Image Search are contained in `data/google-images`. To download all at once, run
```bash
   ./scripts/download_data.sh data/google-images
```
All downloaded images are sub-directories, such as
── weat8
│   ├── attr_man
│   │   ├── aunt_0.jpg
│   │   ├── aunt_1.jpg
│   │   ├── aunt_2.jpg
|	|	...
|	|	└── technology_9.jpg
│   ├── attr_woman
│   │   ├── aunt_0.jpg
│   │   ├── aunt_1.jpg
│   │   ├── aunt_2.jpg
|	|	...
|	|	└── technology_9.jpg
|	└── get.sh

## Download pretrained models
Download the pretrained models for:
* [ViLBERT](https://drive.google.com/drive/folders/1Re0L75uazH3Qrep_aRgtaVelDEz4HV9c)
* [VisualBERT](https://drive.google.com/file/d/1QvivVfRsRF518OQSQNaN7aFk6eQ43vP_/view)
* [VLBert](https://github.com/jackroos/VL-BERT/blob/master/model/pretrained_model/PREPARE_PRETRAINED_MODELS.md): TODO describe which one

LXMert is built on HuggingFace, which handles the model download when creating an instance.

An adapted open-source helper script for downloading models from Google drive: 
```bash
	python scripts/download_gdrive.py GOOGLE_DRIVE_ID PATH_TO_SAVE
```

## Download and extract image features
For the models below, pretrained on either COCO or Conceptual Captions, some of the pre-extracted image features were available in their respective repos. For modeling data from Google Image search, we followed each implementation's original pipeline for feature extraction.
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
	SPLIT=conceptual_image_val

	# create docker image
 	docker build -f caffe/docker/standalone/gpu/Dockerfile -t caffe_image_features .

   	# run container
	docker container run -t -v $BASE_DIR/features:$BASE_DIR/features \
	    --gpus all caffe_image_features python2.7 $BASE_DIR/tools/generate_tsv.py --cfg $BASE_DIR/experiments/cfgs/faster_rcnn_end2end_resnet.yml --def $BASE_DIR/models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt --net $BASE_DIR/models/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel --total_group 1 --group_id 0 --split $SPLIT --gpu 0,1,2,3,4,5,6,7 --out $OUTPUT_FILE --data_dir $DATA_DIR --class-file $CLASS_FILE

    # convert features to LMDB
	sudo chown -R $USER:$USER $(dirname $OUTPUT_FILE)
	python vilbert_beta/scripts/convert_general_lmdb.py --infile_pattern bottom-up-attention/features/conceptual/conceptual_val_resnet101_faster_rcnn_genome.tsv --save_path bottom-up-attention/features/conceptual/conceptual_val_resnet101_faster_rcnn_genome.lmdb

	python vilbert_beta/scripts/convert_general_lmdb.py --infile_pattern bottom-up-attention/features/google-images/angry_black_women_resnet101_faster_rcnn_genome.tsv --save_path bottom-up-attention/features/google-images/angry_black_women_resnet101_faster_rcnn_genome.lmdb/
```

### VLBERT

## LXMert

### VL-BERT
If Docker image has already been created above, then:
```
	# run container
	docker container run -t -v $BASE_DIR/features:$BASE_DIR/features \
	     --gpus all caffe_image_features \
	     python2.7 $BASE_DIR/tools/generate_tsv_v2.py --cfg $BASE_DIR/experiments/cfgs/faster_rcnn_end2end_resnet.yml \
	     	       --def $BASE_DIR/models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt \
		       --net $BASE_DIR/models/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel \
		       --total_group 1 --group_id 0 --split $SPLIT \
		       --gpu 0,1,2,3,4,5,6,7 --out $OUTPUT_FILE --data_dir $DATA_DIR

        # convert features to LMDB
	sudo chown -R $USER:$USER $(dirname $OUTPUT_FILE)
	python vilbert_beta/scripts/convert_general_lmdb.py --infile_pattern bottom-up-attention/features/conceptual/conceptual_val_resnet101_faster_rcnn_genome.tsv --save_path bottom-up-attention/features/conceptual/conceptual_val_resnet101_faster_rcnn_genome.lmdb

	python vilbert_beta/scripts/convert_general_lmdb.py --infile_pattern bottom-up-attention/features/google-images/angry_black_women_resnet101_faster_rcnn_genome.tsv --save_path bottom-up-attention/features/google-images/angry_black_women_resnet101_faster_rcnn_genome.lmdb/
```
```


## Face Detection
To run over custom images,
```bash
    cd scripts
    # TODO forked version
```
# ViLBERT
sudo docker container run -t -v /storage/ccross/bias-grounded-bert/bottom-up-attention/features:/storage/ccross/bias-grounded-bert/bottom-up-attention/features \
     	    	      --gpus all caffe_image_features \
     	    python2.7 /storage/ccross/bias-grounded-bert/bottom-up-attention//tools/generate_tsv.py \
	    	      --cfg /storage/ccross/bias-grounded-bert/bottom-up-attention/experiments/cfgs/faster_rcnn_end2end_resnet.yml \
	     	      --def /storage/ccross/bias-grounded-bert/bottom-up-attention//models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt \
		      --net /storage/ccross/bias-grounded-bert/bottom-up-attention/models/faster_rcnn_models/resnet101_faster_rcnn_final_iter_320000.caffemodel \
		      --total_group 1 --group_id 0 --split  custom \
		      --gpu 0,1,2,3,4,5,6,7 --out /storage/ccross/bias-grounded-bert/bottom-up-attention/features/conceptual/conceptual_val_resnet101_faster_rcnn_genome.tsv \
		      --data_dir /storage/ccross/bias-grounded-bert/bottom-up-attention/data/concap-bias-val \
		      --class_file /storage/ccross/bias-grounded-bert/bottom-up-attention/data/genome/1600-400-20/objects_vocab.txt


sudo docker container run -t -v /storage/ccross/bias-grounded-bert/bottom-up-attention/features:/storage/ccross/bias-grounded-bert/bottom-up-attention/features \
     	    	      --gpus all caffe_image_features \
     	    python2.7 /storage/ccross/bias-grounded-bert/bottom-up-attention//tools/generate_tsv.py \
	    	      --cfg /storage/ccross/bias-grounded-bert/bottom-up-attention/experiments/cfgs/faster_rcnn_end2end_resnet.yml \
	     	      --def /storage/ccross/bias-grounded-bert/bottom-up-attention//models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt \
		      --net /storage/ccross/bias-grounded-bert/bottom-up-attention/models/faster_rcnn_models/resnet101_faster_rcnn_final_iter_320000.caffemodel \
		      --total_group 1 --group_id 0 --split  google_images \
		      --gpu 0,1,2,3,4,5,6,7 --out /storage/ccross/bias-grounded-bert/bottom-up-attention/features/google-images/angry_black_women_val_resnet101_faster_rcnn_genome.tsv \
		      --data_dir /storage/ccross/bias-grounded-bert/bottom-up-attention/data/google-images/angry-black-women \
		      --class_file /storage/ccross/bias-grounded-bert/bottom-up-attention/data/genome/1600-400-20/objects_vocab.txt

# VL-BERT
sudo docker container run -t -v /storage/ccross/bias-grounded-bert/bottom-up-attention/features:/storage/ccross/bias-grounded-bert/bottom-up-attention/features \
     	    	      --gpus all caffe_image_features \
     	    python2.7 /storage/ccross/bias-grounded-bert/bottom-up-attention//tools/generate_tsv_v2.py \
	    	      --cfg /storage/ccross/bias-grounded-bert/bottom-up-attention/experiments/cfgs/faster_rcnn_end2end_resnet.yml \
	     	      --def /storage/ccross/bias-grounded-bert/bottom-up-attention//models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt \
		      --net /storage/ccross/bias-grounded-bert/bottom-up-attention/models/faster_rcnn_models/resnet101_faster_rcnn_final_iter_320000.caffemodel --split  custom \
		      --total_group 1 --group_id 0 --gpu 0,1,2,3,4,5,6,7 --out /storage/ccross/bias-grounded-bert/bottom-up-attention/features/conceptual-v2/conceptual_val_resnet101_faster_rcnn_genome.tsv \
		      --data_dir /storage/ccross/bias-grounded-bert/bottom-up-attention/data/concap-bias-val \
		      --class_file /storage/ccross/bias-grounded-bert/bottom-up-attention/data/genome/1600-400-20/objects_vocab.txt

sudo docker container run -t -v /storage/ccross/bias-grounded-bert/bottom-up-attention/features:/storage/ccross/bias-grounded-bert/bottom-up-attention/features \
     	    	      --gpus all caffe_image_features \
     	    python2.7 /storage/ccross/bias-grounded-bert/bottom-up-attention//tools/generate_tsv_v2.py \
	    	      --cfg /storage/ccross/bias-grounded-bert/bottom-up-attention/experiments/cfgs/faster_rcnn_end2end_resnet.yml \
	     	      --def /storage/ccross/bias-grounded-bert/bottom-up-attention//models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt \
		      --net /storage/ccross/bias-grounded-bert/bottom-up-attention/models/faster_rcnn_models/resnet101_faster_rcnn_final_iter_320000.caffemodel --split  custom \
		      --total_group 1 --group_id 0 --gpu 0,1,2,3,4,5,6,7 --out /storage/ccross/bias-grounded-bert/bottom-up-attention/features/google-images-caffe-v2/weat3.tsv \
		      --data_dir /storage/ccross/bias-grounded-bert/bottom-up-attention/data/google-images/weat3 \
		      --class_file /storage/ccross/bias-grounded-bert/bottom-up-attention/data/genome/1600-400-20/objects_vocab.txt