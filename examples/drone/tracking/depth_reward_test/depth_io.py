# depth_io.py
import os, json, glob
from typing import List, Tuple, Optional
import numpy as np
import imageio.v2 as imageio
import torch

def read_depth_meters(path: str, sidecar_json: Optional[str]=None,
                      assume_max_depth: float=10.0) -> np.ndarray:
    """
    读取一张深度图并还原为 float32 米。
    支持：
      - .npy        : 直接 meters
      - .png 16-bit : 视为毫米（uint16），除以 1000 得米
      - .png 8-bit  : 需要 sidecar json 的 min/max；否则用固定最大深度估计（有误差）
    返回: [H,W] float32 meters (NaN 表示无效，可自行替换)
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        d = np.load(path).astype(np.float32)
        return d

    img = imageio.imread(path)

    # 16-bit PNG（毫米）
    if img.dtype == np.uint16:
        d = img.astype(np.float32) / 1000.0
        # 把 0 当成无效（不少模拟器会用0表示miss）
        d[d <= 0.0] = np.nan
        return d

    # 8-bit PNG（需要min/max）
    if img.dtype == np.uint8:
        if sidecar_json is None:
            # 尝试同名.json
            cand = os.path.splitext(path)[0] + ".json"
            sidecar_json = cand if os.path.exists(cand) else None
        if sidecar_json and os.path.exists(sidecar_json):
            meta = json.load(open(sidecar_json, "r"))
            dmin = float(meta.get("min", 0.0))
            dmax = float(meta.get("max", assume_max_depth))
            scale = max(dmax - dmin, 1e-6)
            d = (img.astype(np.float32) / 255.0) * scale + dmin
            d[d <= 0.0] = np.nan
            return d
        else:
            # 退而求其次：用固定上限估计（会损精度）
            d = (img.astype(np.float32) / 255.0) * float(assume_max_depth)
            d[d <= 0.0] = np.nan
            return d

    # 其他格式：尝试按浮点读取（部分 EXR/TIFF）
    if np.issubdtype(img.dtype, np.floating):
        d = img.astype(np.float32)
        d[d <= 0.0] = np.nan
        return d

    raise ValueError(f"Unsupported depth format: {path} (dtype={img.dtype})")


def load_depth_batch(pattern: str,
                     limit: Optional[int]=None,
                     assume_max_depth: float=10.0) -> torch.Tensor:
    """
    批量加载匹配 pattern 的深度图，返回 [N,H,W] float32 meters (NaN->用max深度替换) 的 torch.Tensor
    pattern 例子: 'data/depth/*.png' 或 'data/depth/*.npy'
    """
    files = sorted(glob.glob(pattern))
    if limit is not None:
        files = files[:limit]
    arrs = []
    H0 = W0 = None
    for f in files:
        d = read_depth_meters(f, assume_max_depth=assume_max_depth)
        if H0 is None:
            H0, W0 = d.shape
        else:
            if d.shape != (H0, W0):
                raise ValueError(f"All depths must share the same size. {f} has {d.shape}, expected {(H0,W0)}")
        # 用 max_depth 替换 NaN，保留一个 mask 的话可以另行返回
        d = np.nan_to_num(d, nan=assume_max_depth)
        arrs.append(d[np.newaxis, ...])
    if not arrs:
        raise FileNotFoundError(f"No files matched: {pattern}")
    batch = np.concatenate(arrs, axis=0).astype(np.float32)  # [N,H,W]
    return torch.from_numpy(batch)


def save_depth_as_png16(path: str, depth_m: np.ndarray):
    """把 float 米保存为 16-bit PNG（毫米）。"""
    mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
    imageio.imwrite(path, mm)
