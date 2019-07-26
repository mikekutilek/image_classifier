# AI Programming Image Classifier
Image Classifier Project for Udacity AI Programming Nanodegree

## Train.py
Trains a neural network on a data set of flower images. Run like so:
```bash
python train.py <data-dir> --save_dir <checkpoint-directory> --arch <model-architecture> --learning_rate <learning-rate> --hidden_units <hidden-layer-units> --epochs <num-epochs> --gpu <gpu enabled or not>
```

## Predict.py
Predicts a flower based on a given image using the trained neural network. Run like so:
```bash
python predict.py <input> <checkpoint> --top_k <top k classes> --category_names <category name json file> --gpu <gpu enabled or not>
```
