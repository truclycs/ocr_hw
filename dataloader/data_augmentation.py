import numpy as np
import imgaug as ia
from PIL import Image
from imgaug import augmenters as iaa


class ImageAugTransform:
    def __init__(self):
        sometimes = self.sometimes()
        self.aug = iaa.Sequential(iaa.SomeOf((1, 5), [
            # blur
            sometimes(iaa.OneOf([iaa.GaussianBlur(sigma=(0, 1.0)), iaa.MotionBlur(k=3)])),
            # color
            sometimes(iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)),
            sometimes(iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6), per_channel=True)),
            sometimes(iaa.Invert(0.25, per_channel=0.5)),
            sometimes(iaa.Solarize(0.5, threshold=(32, 128))),
            sometimes(iaa.Dropout2d(p=0.5)),
            sometimes(iaa.Multiply((0.5, 1.5), per_channel=0.5)),
            sometimes(iaa.Add((-40, 40), per_channel=0.5)),
            sometimes(iaa.JpegCompression(compression=(5, 80))),
            # distort
            sometimes(iaa.Crop(percent=(0.01, 0.05), sample_independently=True)),
            sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.01))),
            sometimes(iaa.Affine(scale=(0.7, 1.3), translate_percent=(-0.1, 0.1),
                                 order=[0, 1], cval=(0, 255), mode=ia.ALL)),
            sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.01))),
            sometimes(iaa.OneOf([iaa.Dropout(p=(0, 0.1)),
                                 iaa.CoarseDropout(p=(0, 0.1), size_percent=(0.02, 0.25))])), ],
            random_order=True), random_order=True)

    def sometimes(self):
        return lambda aug: iaa.Sometimes(0.3, aug)

    def __call__(self, image):
        image = np.array(image)
        image = self.aug.augment_image(image)
        image = Image.fromarray(image)
        return image
