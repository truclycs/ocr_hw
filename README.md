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
<!-- ```
flame/data
	|
	├── coco_dataset.py
	├── pascal_dataset.py
	├── labelme_dataset.py
	└── altheia_dataset.py

configs/
	|
	├── birdview_vehicles_training.yaml
	├── birdview_vehicles_testing.yaml
	├── coco_training.yaml
	├── coco_testing.yaml
	├── publaynet_training.yaml
	├── publaynet_testing.yaml
	├── pascal_training.yaml
	└── pascal_testing.yaml
``` -->

### 2.3 Download
<!-- * COCO Train/Val/Test 2017
```bash
https://cocodataset.org/#download
```

* Pascal VOC 2007
```bash
http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
```
```bash
http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
```

* Pascal VOC 2012
```bash
http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
```

* Publaynet
```bash
https://developer.ibm.com/exchanges/data/all/publaynet/
``` -->

### 2.4 Dataset Stats
<!-- |ID|Dataset Name|Train|Val|Test|Format|
|:--:|:--------:|:--------:|:--:|:--:|:--:|
1|COCO 2017 |118,287|5,000|-|COCO JSON|
2|Pascal VOC 2007 |5,011|4,952|-|PASCAL XML|
3|Pascal VOC 2012 |1,464|1,449|-|PASCAL XML|
4|PubLayNet |335,703|11,245|11,405|COCO JSON| -->


## 3. Pretrained Weights
<!-- ### 3.1 Weights Location of Project
* Create Weight Folder
```python
mkdir checkpoint/efficientnet_pretrained_weight
mkdir checkpoint/efficientdet_pretrained_weight
```
* Download Weight
```bash
!wget <weight-path> -O checkpoint/efficientdet_pretrained_weight/<weight-path-name>
```
```bash
!wget <weight-path> -O checkpoint/efficientnet_pretrained_weight/<weight-path-name>
```

### 3.2 EfficientNet Pretrained Weights
```bash
https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth
https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b1-f1951068.pth
https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b2-8bb594d6.pth
https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b3-5fb5a3c3.pth
https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth
https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b5-b6417697.pth
https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b6-c76e70fd.pth
https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth
```

### 3.3 EfficientDet Pretrained Weights
```bash
https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/releases/download/1.0/efficientdet-d0.pth
https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/releases/download/1.0/efficientdet-d1.pth
https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/releases/download/1.0/efficientdet-d2.pth
https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/releases/download/1.0/efficientdet-d3.pth
https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/releases/download/1.0/efficientdet-d4.pth
https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/releases/download/1.0/efficientdet-d5.pth
https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/releases/download/1.0/efficientdet-d6.pth
https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/releases/download/1.0/efficientdet-d7.pth
https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/releases/download/1.0/efficientdet-d8.pth
``` -->

## 4. Usage
### 4.1 Todo
<Udating>
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

## 5. Performance
<Updating>

## 6. Explaination
<Updating>