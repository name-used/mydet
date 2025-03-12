import json
from config import patch_size

# 这个是自定义的数据集模式
with open(rf'./source.lib') as f:
    source_lib = json.load(f)

# source_train = [source_lib['data'][name] for name in source_lib['group']['train']]
# source_valid = [source_lib['data'][name] for name in source_lib['group']['valid']]
# source_test = [source_lib['data'][name] for name in source_lib['group']['test']]

source_lib['group']['train'] = source_lib['group']['train'][:1]
source_lib['group']['valid'] = source_lib['group']['valid'][:1]

train = dict(
    source=[source_lib['data'][name] for name in source_lib['group']['train']],
    base_zoom=1.,
    patch_size=patch_size,
    use_bbox=True,
    use_mask=False,
    use_seg=False,
)

valid = dict(
    source=[source_lib['data'][name] for name in source_lib['group']['valid']],
    base_zoom=1.,
    patch_size=patch_size,
    use_bbox=True,
    use_mask=False,
    use_seg=False,
)
