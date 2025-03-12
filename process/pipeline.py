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


# # 原本的增强
# self.transformer = A.Compose(
#     [
#         # A.RandomResizedCrop(height=size, width=size, scale=(0.8, 1.0), ratio=(0.9, 1.11), p=0.0),
#         A.Blur(p=0.01),
#         A.MedianBlur(p=0.01),
#         A.ToGray(p=0.01),
#         A.CLAHE(p=0.01),
#         A.RandomBrightnessContrast(p=0.0),
#         A.RandomGamma(p=0.0),
#         A.ImageCompression(quality_lower=75, p=0.0)
#     ]
# )
#
#
# def train_albumentation(self, image: np.ndarray, label: np.ndarray):
#     image = image.copy()
#     label = label.copy()
#
#     # 颜色增强
#     image = self.trans_color(image)
#
#     # 空间贴图
#     image, label = self.trans_space_cv(image, label)
#
#     # 空间八状态增强
#     image, label = self.trans_space_8state(image, label)
#
#     # 随机丢弃，原代码就是注释状态，暂时不管
#     # labels = cutout(img, labels, p=0.5)
#     # nl = len(labels)  # update after cutout
#
#     image = np.ascontiguousarray(image)
#     label = np.ascontiguousarray(label)
#     return image, label
#
#
# def trans_color(self, image: np.ndarray):
#     # 放弃对原数据处理操作的思考，因为原代码写的太垃了。。。
#     image = self.transformer(image=image)['image']
#     # 瞅瞅这代码写的，albumentations 里面本来就有 hsv 增强，不用，非要自己写一个
#     # 写也就算了吧，明明已经自己写了一个基于 albumentations 包装的 Albumentations
#     # 这功能逻辑完全一样的增强为啥不写到包装类里，反而用 C++ 风格单起了一个函数？？？
#     # 编程逻辑实在是混乱不堪。。。
#     image = np.ascontiguousarray(image)
#     augment_hsv(image, hgain=0.015, sgain=0.7, vgain=0.4)
#     return image
#
#
# def trans_space_cv(self, image: np.ndarray, label: np.ndarray):
#     # 如果轮廓已经太多，或者随机概率没投到，就不增强了
#     if len(label) > 30 or random.random() > 1:
#         return image, label
#
#     image = image.copy()
#     label = lurd2xywh(label)
#     # 640 * 640
#     H, W, _ = image.shape
#
#     # 随机添加若干个轮廓
#     for _ in range(np.random.randint(10)):
#         # 从图库中随机抽取一组标签
#         index = np.random.randint(len(self.labels))
#         if len(self.labels[index]) <= 0:
#             continue
#         tip_lb = self.labels[index][np.random.randint(len(self.labels[index]))]
#
#         # 把对应图块扣下来
#         bd = np.random.randint(7)
#         l, u, r, d = tip_lb[1:].astype(int)
#         tip_im = self.images[index][u - bd: d + bd, l - bd: r + bd, :]
#
#         # 在目标图中找到一块相对干净的区域把标签贴上去
#         # 采用随机法, 限制随机次数 < 20
#         h, w, _ = tip_im.shape
#
#         rects = label[:, 1:].T
#         for test in range(20):
#             x = np.random.randint(w // 2, W - w // 2)
#             y = np.random.randint(h // 2, H - h // 2)
#             if ((abs(rects[0] - x) < rects[2] + w) & (abs(rects[1] - y) < rects[3] + h)).any():
#                 # print('failed')
#                 continue
#             # print('success')
#             l = x - w // 2
#             u = y - h // 2
#             r = l + w
#             d = u + h
#             # cv2.rectangle(image, (l, u), (r, d), [255, 0, 0], 7)
#             image[u: d, l: r, :] = tip_im
#             label = np.concatenate([label, [(0, x, y, w - 2 * bd, h - 2 * bd)]])
#             # label = np.concatenate([label, np.expand_dims(tip_lb, axis=0)])
#             break
#     label = xywh2lurd(label)
#     return image, label
#
#
#     @staticmethod
#     def trans_space_8state(image: np.ndarray, label: np.ndarray):
#         h, w, _ = image.shape
#         assert h == w
#         # 在 (x, y, w, h)::[0, 1] 上计算
#         label = lurd2xywh(label, 1 / w)
#         # 原代码的空间增强也完全没思考过样本空间的问题，基本的八状态都没反映出来
#         # 图像随机逆时针旋转 0-3 次
#         k = random.randint(0, 3)
#         image = np.rot90(image, k=k)
#         # 每次逆时针旋转横纵轴交换，符号交换，交换算法如下：
#         # 令 sx 表示 x 坐标符号，sy 表示 y 坐标符号，k 表示旋转次数
#         # 符号计算如下：
#         # sx' = (sx * sy * -1)**k * (-1)**(k(k-1)/2)
#         # sy' = (sx * sy)**k * (-1)**(k(k-1)/2)
#         # 数字计算如下：
#         # x', y' = [x, y][::(-1) ** k]
#         # 特别需要注意的是，数学坐标轴与图像坐标轴存在镜像关系，图像的逆时针旋转对应坐标轴顺时针旋转，此时 sx、xy 的 -1 需要交换
#         # 另一个需要注意的问题是，数学运算和代码逻辑是不同的，代码上需要先交换数值再改符号
#         # 按照这个逻辑，会发现逆时针一直改 x，顺时针一直改 y 即可
#         label[:, 1:3] -= 0.5
#         # signs = np.sign(label[:, 1:3]).astype(int)
#         # temp = signs.copy()
#         # signs[:, 0] = (temp[:, 0] * temp[:, 1]) ** k * (-1)**(k*(k-1)//2)
#         # signs[:, 1] = (temp[:, 0] * temp[:, 1] * -1) ** k * (-1)**(k*(k-1)//2)
#         # 奇数次旋转会导致 x-y 坐标交换
#         for _ in range(k):
#             temp = label.copy()
#             label[:, 1] = temp[:, 2]
#             label[:, 2] = temp[:, 1]
#             label[:, 3] = temp[:, 4]
#             label[:, 4] = temp[:, 3]
#             # label[:, 1:3] *= signs
#             label[:, 2] *= -1
#         label[:, 1:3] += 0.5
#         # 镜像或者不镜像（水平镜像、垂直镜像、主对角线镜像，每一个同四向旋转都构成完整的均匀状态空间，因此此处 flip、fliplr、flipud 效果完全一致）
#         # 考虑到标签处理的复杂性，此处选择水平对称
#         if random.random() < 0.5:
#             image = np.fliplr(image)
#             label[:, 1] = 1 - label[:, 1]
#         label = xywh2lurd(label, w)
#         return image, label
#
#     def data_format(self, image: np.ndarray, label: np.ndarray):
#         labels_out = torch.zeros((len(label), 6))
#         if len(labels_out):
#             label = lurd2xywh(label, 1 / self.target_size)
#             labels_out[:, 1:] = torch.from_numpy(label)
#         # Convert
#         image = image.transpose((2, 0, 1))
#         image = np.ascontiguousarray(image)
#         return torch.from_numpy(image), labels_out
#
#     def lurd2xywh(label: np.ndarray, value_zoom: float = 1.):
#         label = label.copy()
#         label[:, 3: 5] = label[:, 3: 5] - label[:, 1: 3]
#         label[:, 1: 3] = label[:, 1: 3] + label[:, 3: 5] / 2
#         label[:, 1:] *= value_zoom
#         return label
#
#     def xywh2lurd(label: np.ndarray, value_zoom: float = 1.):
#         label = label.copy()
#         label[:, 1: 3] = label[:, 1: 3] - label[:, 3: 5] / 2
#         label[:, 3: 5] = label[:, 1: 3] + label[:, 3: 5]
#         label[:, 1:] *= value_zoom
#         return label
