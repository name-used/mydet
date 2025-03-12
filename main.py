import os
import shutil
from pathlib import Path
import sys


def main():
    # 运行时配置（性能无关）
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    work_root = Path(rf'D:\jassor_work_output\basic_detector')
    current_root = Path(os.path.abspath(os.curdir))
    base_root = current_root
    os.makedirs(work_root, exist_ok=True)

    # 文件配置（性能相关）
    shutil.copy(base_root / rf'config.py', work_root / rf'config.py')
    shutil.copy(base_root / rf'source.lib', work_root / rf'source.lib')
    shutil.copytree(base_root / rf'process', work_root / rf'process', dirs_exist_ok=True)
    shutil.copytree(base_root / rf'configs', work_root / rf'configs', dirs_exist_ok=True)
    # 无聊（没必要搬，但搬过去好看）
    shutil.copy(base_root / rf'main.py', work_root / rf'main.py')
    shutil.copy(base_root / rf'build.py', work_root / rf'build.py')

    # 重定向运行时环境至 base_root，这样就可以直接执行目标路径下的配置和代码了，可用于访问训练时代码环境
    sys.path = [p for p in sys.path if os.path.abspath(p) != str(current_root)]
    sys.path.append(str(base_root))
    os.chdir(str(base_root))
    sys.argv = []

    # 执行训练方法
    from process import train
    # work_root 是最终文件的输出目录
    train(work_root)


if __name__ == '__main__':
    main()
