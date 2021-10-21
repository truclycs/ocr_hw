## 1. References
[1] Attention Is All You Need: https://arxiv.org/abs/1706.03762 \
[2] Github: https://github.com/pbcquoc/vietocr

## 2. Dataset
### 2.1 Todo
<!-- - [x] supporting for COCO 2017, PubLayNet dataset with COCO format.
- [x] supporting for PASCAL VOC 2007, 2012 dataset with XML format.
- [x] supporting for dataset with LABELME format.
- [x] supporting for dataset with ALTHEIA format. -->

### 2.2 Structure of Configs
```
dataset/
    |
    ├── annotation/
    |       |
    |       └──cin/
    |           |
    │           ├── train.txt
    |           ├── val.txt
    │           └── test.txt
    └── images/
            |
            ├── image1.jpg
            ├── image2.jpg
            ├── ...
            └── imagen.jpg

config/
    |
    |── vgg_seq2seq.yml
    └── vgg_transformer.yml
```

### 2.3 Download
* Cinnamon: Handwriting OCR for Vietnamese Address
```bash
https://drive.google.com/drive/folders/1Qa2YA6w6V5MaNV-qxqhsHHoYFRK5JB39
```

## 3. Pretrained Weights


## 4. Usage
### 4.1 Todo
- [ ] Predicting with batch images.
- [ ] Experience with other backbones (current: only VGG19).
<!-- - [x] Applied for many dataset format included coco, pascal, labelme, altheia.
- [x] Applied **imgaug** for augmenting data, dataloader with setting 'num_workers', 'pin_memory', 'drop_last' for optimizing training.
- [x] Rearraged training and testing flow with Ignite Pytorch.
- [x] Refactored **Focal Loss** and **mAP** for training and evaluation.
- [x] Applied **region_predictor** function for visualizing predicted results.
- [ ] Updating FP16 (automatic mixed precision), DDP (DistributedDataParallel) for faster training on GPUs.
- [ ] Updating Tensorboard, Profiler. -->

### 4.2 Usage
* Training
```python
CUDA_VISIBLE_DEVICES=<cuda_indice> python train.py --config config/vgg_transformer.yml
```

* Testing
```python
CUDA_VISIBLE_DEVICES=<cuda_indice> python test.py --config config/vgg_transformer.yml
```

* Predicting
```python
CUDA_VISIBLE_DEVICES=<cuda_indice> python predict.py --config config/vgg_transformer.yml --image <image_path>
```


## 5. Performance
<Updating>

## 6. Explaination
<Updating>