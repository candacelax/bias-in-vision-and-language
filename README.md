This is the repo for our paper "Biases in Joint Visual and Language Embeddings". We implement a version of WEAT/SEAT for visually grounded word embeddings. This is code borrowed and modified from the [SEAT paper](https://github.com/W4ngatang/sent-bias).

## Installation
Create the conda environment.
```bash
git clone git@github.com:candacelax/vilbert_beta.git # forked
git clone git@github.com:candacelax/visualbert.git # forked

git clone git@github.com:candacelax/bias-grounded-bert.git
cd bias-grounded-bert
ln -s PATH_TO_VISUALBERT visualbert
ln -s PATH_TO_VILBERT vilbert_beta

conda env create -f environment.yml
python -m spacy download en

# add VisualBERT and ViLBERT to Python paths (consider doing in .bashrc)
export PYTHONPATH="$PYTHONPATH:XX/bias-grounded-bert/visualbert"
export PYTHONPATH="$PYTHONPATH:XX/bias-grounded-bert/vilbert_beta"
export PYTORCH_PRETRAINED_BERT_CACHE="XX/bias-grounded-bert/.pytorch_pretrained_bert"
```


Download the pretrained models for [ViLBERT](https://drive.google.com/drive/folders/1Re0L75uazH3Qrep_aRgtaVelDEz4HV9c) and [VisualBERT](https://drive.google.com/file/d/1QvivVfRsRF518OQSQNaN7aFk6eQ43vP_/view). Save each in the model's respective pretrained-models directory.


## Usage
`./main.py --config CONFIG_FILEPATH`



## Feature Extraction
If you want to run over custom images, you'll need to compute features.
### VisualBert
To compute image features, we use the same model backbone/size as VisualBERT. Detectron model id is 137851257 ([see Model Zoo](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md)).

```bash
./scripts/image-features/extract_image_features.sh -m MODEL_NAME -d IMAGE_DIR -o FEATURE_PATH
```
where
* MODEL_NAME is either 'visualbert' or 'vilbert'
* IMAGE_DIR is directory of bias test images (e.g. data/google-images/weat6)
* FEATURE_PATH is location to save features (e.g. visualbert/image-features/google-images/weat6_features.th)