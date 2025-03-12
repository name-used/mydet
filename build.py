import json
import os
from pathlib import Path
from typing import Dict

import imagesize
import jassor.utils as J

import config


def main():
    source_root = Path(rf'./resources')
    with open(source_root / rf'key_map.json') as f:
        key_map: Dict[str, str] = json.load(f)

    source_lib = {
        'group': {'train': [], 'valid': [], 'test': []},
        'data': {},
    }
    for name, group in key_map.items():
        image_path = source_root / 'images' / rf'{name}.jpg'
        bbox_path = source_root / 'bboxes' / rf'{name}.json'
        if not os.path.exists(image_path): continue
        if not os.path.exists(bbox_path): continue
        group = key_map[name]
        source_lib['group'][group].append(name)
        w, h = imagesize.get(image_path)
        source_lib['data'][name] = {
            'name': name,
            'size': (w, h),
            'zoom': 1,
            'times': round(w * h / config.patch_size ** 2),
            'image_path': image_path.__fspath__(),
            'bbox_path': bbox_path.__fspath__(),
        }

    source_lib['group']['valid'] = source_lib['group']['train']
    with open('./source.lib', 'w') as f:
        json.dump(source_lib, f, cls=J.JassorJsonEncoder)


if __name__ == '__main__':
    main()
