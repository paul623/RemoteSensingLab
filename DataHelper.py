"""
@Project ：RemoteSensingLab 
@File    ：DataHelper.py
@IDE     ：PyCharm 
@Author  ：paul623
@Date    ：2024/4/28 13:48 
"""
import os
from pathlib import Path


def get_pair_path(directory_name, root):
    ref_label, pred_label = directory_name.split('-')
    ref_tokens, pred_tokens = ref_label.split('_'), pred_label.split('_')
    paths = [None] * 4  # 索引 0, 1, 2 分别对应 pred_modis, ref_lansat, pred_landsat
    # 使用 Path 对象处理路径
    directory_path = Path(os.path.join(root, directory_name))

    # 使用 glob 查找 .tif 文件
    for f in directory_path.glob('*.tif'):
        if f.match(f"*{ref_tokens[0]}{ref_tokens[1]}*"):
            paths[0] = f.resolve().__str__()
        elif f.match(f"*{ref_tokens[0]}{ref_tokens[2]}*"):
            paths[1] = f.resolve().__str__()
        elif f.match(f"*{pred_tokens[0]}{pred_tokens[1]}*"):
            paths[2] = f.resolve().__str__()
        elif f.match(f"*{pred_tokens[0]}{pred_tokens[2]}*"):
            paths[3] = f.resolve().__str__()

    # 检查 paths 是否包含 None，确保所有路径都被找到
    if None in paths:
        raise FileNotFoundError("Not all expected files were found in the directory.")

    return paths


def getDataLoader(option):
    list = []
    names = []
    assert option in ["LGC", "CIA"]
    if option == "LGC":
        root = Path("/home/zbl/datasets_paper/LGC/val/")
    else:
        root = Path("/home/zbl/datasets_paper/CIA/val/")
    for path in os.listdir(root):
        list.append(get_pair_path(path, root))
        names.append(path)
    return list, names    # return m1,f1,m2,f2 2 is target
