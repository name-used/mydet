from typing import Dict, Optional, Union, Tuple, List
import cv2
from mmcv.transforms import BaseTransform
from mmengine.registry import TRANSFORMS
import albumentations as A
import numpy as np


@TRANSFORMS.register_module()
class AlbumentationsTransform(BaseTransform):
    def __init__(self):
        """Wrap Albumentations transforms to make it compatible with mmdet pipeline."""
        self.transforms = A.Compose([
            # A.RandomResizedCrop(height=size, width=size, scale=(0.8, 1.0), ratio=(0.9, 1.11), p=0.0),
            A.Blur(p=0.01),
            A.MedianBlur(p=0.01),
            A.ToGray(p=0.01),
            A.CLAHE(p=0.01),
            A.RandomBrightnessContrast(p=0.0),
            A.RandomGamma(p=0.0),
            A.ImageCompression(quality_lower=75, p=0.0)
        ])

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        """Apply the Albumentations transforms to the image."""
        # Extract the image from results
        image = results['img']

        # 颜色增强
        image = self.transforms(image=image)['image']
        image = np.ascontiguousarray(image)
        augment_hsv(image, hgain=0.015, sgain=0.7, vgain=0.4)

        # Update the results
        results['img'] = image
        return results


def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    # HSV color-space augmentation
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed


@TRANSFORMS.register_module()
class Identify(BaseTransform):
    def __init__(self):
        pass

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        return results
