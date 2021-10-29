## 1. References
[1] Attention Is All You Need: https://arxiv.org/abs/1706.03762 \
[2] Github: https://github.com/pbcquoc/vietocr

## 2. Dataset
### 2.1 Todo

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

* HANDS-VNOnDB: Vietnamese Online Handwriting Database
```bash
http://tc11.cvc.uab.es/datasets/HANDS-VNOnDB2018_1/
```

## 3. Pretrained Weights


## 4. Usage
### 4.1 Todo
- [ ] Predicting with batch images.
- [ ] Experience with other backbones (current: VGG19, ResNet50).

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
