This is the repo for our paper "Biases in Joint Visual and Language Embeddings". We implement a version of WEAT/SEAT for visually grounded word embeddings. This is code borrowed and modified from the [SEAT paper](https://github.com/W4ngatang/sent-bias).

## Installation
Create the conda environment.
```bash
git clone git@github.com:candacelax/bias-grounded-bert.git
cd bias-grounded-bert
conda env create -f environment.yml
python -m spacy download en
```


Download the pretrained models for [ViLBERT](https://drive.google.com/drive/folders/1Re0L75uazH3Qrep_aRgtaVelDEz4HV9c) and [VisualBERT](https://drive.google.com/file/d/1QvivVfRsRF518OQSQNaN7aFk6eQ43vP_/view). Save each in the model's respective pretrained-models directory.


## Usage
`./main.py --config CONFIG_FILEPATH`



## Installation
`
pip install spacy==2.1.0
python -m spacy download en
pip install neuralcoref --no-binary neuralcoref

for ViLBERT, we removed the python-prctl from requirements.txt

`
Update PYTORCH_PRETRAINED_BERT_CACHE environment variable.

## Feature Extraction
If you want to run over custom images, you'll need to compute features.
### VisualBert
To compute image features, we use the same model backbone/size as VisualBERT. Detectron model id is 137851257 ([see Model Zoo](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md)).

```bash
./scripts/image-features/extract_image_features.sh MODEL_NAME IMAGE_DIR FEATURE_PATH
```
where
* MODEL_NAME is either 'visualbert' or 'vilbert'
* IMAGE_DIR is directory of bias test images (e.g. data/google-images/weat6)
* FEATURE_PATH is location to save features (e.g. visualbert/image-features/google-images/weat6_features.th)


### ViLBERT


For running ViLBERT:
`
conda create -n vilbert python=3.6
conda activate vilbert
cd vilbert_beta
pip install -r requirements.txt
`

For extracting visual features:
`conda install caffe`


For running VisualBERT:
`
conda create -n visual-bias python=3.7
conda activate visual-bias
cd vilbert_beta

conda install numpy pyyaml setuptools cmake cffi tqdm pyyaml scipy ipython mkl mkl-include cython typing h5py pandas nltk spacy numpydoc scikit-learn jpeg tensorflow
pip install tensorflow-hub

#Please check your cuda version using `nvcc --version` and make sure the cuda version matches the cudatoolkit version.
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

pip install -r allennlp-requirements.txt
pip install --no-deps allennlp==0.8.0
python -m spacy download en_core_web_sm
pip install attrdict
pip install pycocotools
pip install commentjson
`