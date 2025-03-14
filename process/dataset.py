from torch.utils.data.dataset import Dataset as DatasetInterface
import json
import random
from typing import List, Dict, Any
import numpy as np
import jassor.components.data as D
import jassor.shape as S
import jassor.utils as J
from mmcv.transforms import Compose


class Dataset(DatasetInterface):
    def __init__(self, source: List[Dict[str, Any]], base_zoom: float, patch_size: int, pipeline: list, use_bbox: bool = True, use_mask: bool = True, use_seg: bool = True):
        self.sample_basics = []
        self.images = []
        self.bboxes = []
        self.masks = []
        self.segs = []
        self.patch_size = patch_size
        self.source = source
        self.base_zoom = base_zoom

        for data in source:
            self.images.append(D.load(data['image_path']))
            # 图像内建信息
            self.sample_basics.append({
                'times': data['times'],
                'size': data['size'],
                'zoom': data['zoom'],
            })
            # 三个 head 头
            if use_bbox:
                with open(data['bbox_path']) as f:
                    # bbox [(type, (l, u, r, d))]:: [0, w/h]
                    temp = json.load(f)
                # t, bbox = zip(*temp) if temp else ([], [])
                # bbox = np.concatenate([t, bbox], axis=1) if bbox else np.zeros(shape=(0, 5), dtype=np.int64)
                bbox = [(t, l, u, r, d) for t, (l, u, r, d) in temp]
                bbox = np.asarray(bbox) if bbox else np.zeros(shape=(0, 5), dtype=np.int64)
                self.bboxes.append(bbox)

            if use_mask:
                with open(data['mask_path']) as f:
                    # mask [(type, [(x, y)]]:: [0, w/h]
                    mask = json.load(f)
                    mask = [(t, S.SimplePolygon(outer=m)) for t, m in mask]
                self.masks.append(mask)

            if use_seg:
                # seg matrix:: type
                self.segs.append(D.load(data['seg_path']))

        # samples 存储采样坐标
        self.samples = []
        # mmdet 框架转接的预处理管道
        self.pipeline = Compose(pipeline)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item: int):
        i, (l, u, r, d) = self.samples[item]
        returns = dict(image=self.load_image(i, l, u, r, d))
        if self.bboxes: returns['bbox'] = self.load_bbox(i, l, u, r, d)
        if self.masks: returns['mask'] = self.load_mask(i, l, u, r, d)
        if self.segs: returns['seg'] = self.load_seg(i, l, u, r, d)
        return self.transform_as_mmdet(returns)

    def sample_random(self):
        self.samples = []
        for i, basic in enumerate(self.sample_basics):
            w, h = basic['size']
            for _ in range(basic['times']):
                # 随机缩放采样
                random_zoom = 0.9 + 0.3 * random.random()
                # 采集尺寸
                k = round(self.patch_size * self.base_zoom * basic['zoom'] * random_zoom)
                w = max(w, k)
                h = max(h, k)
                # 随机坐标
                left = round(random.random() * (w - k))
                up = round(random.random() * (h - k))
                self.samples.append((i, (left, up, left+k, up+k)))
        random.shuffle(self.samples)

    def sample_by_step(self):
        self.samples = []
        for i, basic in enumerate(self.sample_basics):
            w, h = basic['size']
            # 采集尺寸
            k = round(self.patch_size * self.base_zoom * basic['zoom'])
            for up in J.uniform_iter(h, k, k):
                for left in J.uniform_iter(w, k, k):
                    self.samples.append((i, (left, up, left+k, up+k)))

    def load_image(self, i: int, left: int, up: int, right: int, down: int):
        image = self.images[i]
        patch = image.region(0, left, up, right, down)
        return patch

    def load_bbox(self, i: int, left: int, up: int, right: int, down: int) -> dict:
        bbox = self.bboxes[i].copy()

        # box area
        l, u, r, d = left, up, right, down

        # choose in box
        ignore = 5
        bbox[bbox[:, 1] >= r - ignore, 1] = -999999  # bbox at box right
        bbox[bbox[:, 2] >= d - ignore, 2] = -999999  # bbox at box down
        bbox[bbox[:, 3] <= l + ignore, 3] = -999999  # bbox at box left
        bbox[bbox[:, 4] <= u + ignore, 4] = -999999  # bbox at box up
        bbox = bbox[bbox.sum(axis=1) > 0, :]

        # limit to box
        bbox[bbox[:, 1] < l, 1] = l
        bbox[bbox[:, 2] < u, 2] = u
        bbox[bbox[:, 3] > r, 3] = r
        bbox[bbox[:, 4] > d, 4] = d

        # absolute -> relative
        # [t, l, u, r, d] :: [0 - w/h]
        bbox[:, (1, 3)] -= l
        bbox[:, (2, 4)] -= u

        t = bbox[:, 0].tolist()
        bbox = bbox[:, 1:].astype(np.float32).tolist()
        return dict(type=t, bbox=bbox)

    def load_mask(self, *args, **kwargs) -> dict:
        raise NotImplemented

    def load_seg(self, *args, **kwargs) -> dict:
        raise NotImplemented

    def transform_as_mmdet(self, my_return: dict) -> dict:
        data_samples = {'img_id': 0, 'img': my_return['image'], 'instances': []}
        for t, bbox in zip(my_return['bbox']['type'], my_return['bbox']['bbox']):
            data_samples['instances'].append({
                'bbox': bbox,
                'bbox_label': t,
                'ignore_flag': 0,
            })
        data_samples = self.pipeline(data_samples)
        return data_samples
