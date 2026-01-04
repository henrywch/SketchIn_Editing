# -*- coding: utf-8 -*-
"""
FFD-based sketch deformation for full image.

使用FFD (Free-Form Deformation) 对整张图片进行变形。
"""

import os
import cv2
import math
import numpy as np
from typing import Tuple, Optional, Dict


SPACING     = 32                  # 控制点间距（像素）；小=细腻但更慢
SIGMA       = 10.0                # 控制点位移强度（像素，std）
MOVE_PROB   = 0.99                # 控制点被挪动的概率（降低=更稀疏/更局部）
BINARIZE_OUTPUT = True            # 输出二值化（适合边缘图）


def _bspline_basis(u: np.ndarray):
    """Cubic uniform B-spline basis values for u in [0,1]."""
    B0 = ((1 - u) ** 3) / 6.0
    B1 = (3 * u**3 - 6 * u**2 + 4) / 6.0
    B2 = (-3 * u**3 + 3 * u**2 + 3 * u + 1) / 6.0
    B3 = (u**3) / 6.0
    return [B0, B1, B2, B3]

def _imread_image(path: str) -> np.ndarray:
    """Read image; keep channels (gray or 3-channel)."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] in (3, 4):
        if img.shape[2] == 4:
            img = img[:, :, :3]
        return img
    raise ValueError(f"Unexpected image shape: {img.shape}")

# =========================
# ====   FFD core     =====
# =========================

def _control_points_displacement(
    h: int, w: int, spacing: int, sigma: float,
    move_prob: float, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对全图所有控制点按概率随机位移。
    """
    Ncx = int(math.ceil(w / spacing))
    Ncy = int(math.ceil(h / spacing))
    ctrl = np.zeros((Ncy + 3, Ncx + 3, 2), dtype=np.float32)

    grid_x = (np.arange(Ncx + 3, dtype=np.float32) - 1.0) * spacing
    grid_y = (np.arange(Ncy + 3, dtype=np.float32) - 1.0) * spacing

    for q in range(Ncy + 3):
        for p in range(Ncx + 3):
            if rng.random() < move_prob:
                dx = rng.normal(0.0, sigma)
                dy = rng.normal(0.0, sigma)
                ctrl[q, p, 0] = dx
                ctrl[q, p, 1] = dy

    return ctrl, grid_x, grid_y

def _build_displacement_field(
    h: int, w: int, spacing: int,
    ctrl: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32),
                         np.arange(w, dtype=np.float32),
                         indexing='ij')
    gx = xx / spacing
    gy = yy / spacing
    ix = np.floor(gx).astype(np.int32)
    iy = np.floor(gy).astype(np.int32)
    u = gx - ix
    v = gy - iy
    Bx = _bspline_basis(u)
    By = _bspline_basis(v)

    Ncx = len(grid_x) - 3
    Ncy = len(grid_y) - 3
    dx = np.zeros((h, w), dtype=np.float32)
    dy = np.zeros((h, w), dtype=np.float32)

    for l in range(4):
        for m in range(4):
            px = np.clip(ix + l, 0, Ncx + 2)
            py = np.clip(iy + m, 0, Ncy + 2)
            wlm = Bx[l] * By[m]
            disp = ctrl[py, px]
            dx += wlm * disp[..., 0]
            dy += wlm * disp[..., 1]

    return dx, dy

def _remap_image(img: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        warped = cv2.remap(img, map_x, map_y,
                           interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        return warped
    elif img.ndim == 3 and img.shape[2] == 3:
        chs = []
        for c in range(3):
            ch = cv2.remap(img[..., c], map_x, map_y,
                           interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=255)
            chs.append(ch)
        return np.stack(chs, axis=-1)
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

def deform(
    img: np.ndarray,
    spacing: int = 64,
    sigma: float = 12.0,
    move_prob: float = 0.85,
    seed: Optional[int] = None,
    binarize_output: bool = True,
    return_debug: bool = False
) -> Tuple[np.ndarray, Optional[Dict]]:
    """
    使用FFD对整张图片进行变形。
    
    Args:
        img: 输入图像（灰度或RGB）
        spacing: 控制点间距（像素）
        sigma: 控制点位移强度（像素，std）
        move_prob: 控制点被挪动的概率
        seed: 随机种子
        binarize_output: 是否对输出进行二值化
        return_debug: 是否返回调试信息
    
    Returns:
        变形后的图像，以及可选的调试信息字典
    """
    h, w = img.shape[:2]
    rng = np.random.default_rng(seed)

    ctrl, grid_x, grid_y = _control_points_displacement(
        h, w, spacing, sigma, move_prob, rng
    )

    dx, dy = _build_displacement_field(h, w, spacing, ctrl, grid_x, grid_y)

    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32),
                         np.arange(w, dtype=np.float32),
                         indexing='ij')
    map_x = xx - dx
    map_y = yy - dy

    warped = _remap_image(img, map_x, map_y)

    out = warped
    if binarize_output:
        if img.ndim == 2:
            _, out = cv2.threshold(out, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        else:
            chs = []
            for c in range(3):
                _, ch = cv2.threshold(out[..., c], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                chs.append(ch)
            out = np.stack(chs, axis=-1)

    if return_debug:
        dbg = {
            "warped": warped,
            "dx": dx,
            "dy": dy,
        }
        return out, dbg
    else:
        return out, None


def main():
    """主函数：加载sketch图片，进行变形，并可视化结果。"""
    # 配置输入图片路径（可以修改为你的图片路径）
    sketch_path = "L0_sample2895.png"
    
    # 如果默认路径不存在，尝试其他可能的路径
    if not os.path.exists(sketch_path):
        # 尝试当前目录
        if os.path.exists("000000000724_sketch.png"):
            sketch_path = "000000000724_sketch.png"
        else:
            print(f"错误：找不到图片文件。请修改 main() 中的 sketch_path 变量。")
            print(f"尝试的路径: {sketch_path}")
            return
    
    print(f"加载图片: {sketch_path}")
    img = _imread_image(sketch_path)
    print(f"图片尺寸: {img.shape}")
    
    # 使用全局配置参数进行变形
    print("开始变形...")
    result, _ = deform(
        img,
        spacing=SPACING,
        sigma=SIGMA,
        move_prob=MOVE_PROB,
        seed=None,  # 可以设置为固定值以得到可重复的结果
        binarize_output=BINARIZE_OUTPUT,
        return_debug=False
    )
    print("变形完成！")
    
    # 创建可视化：并排显示原图和结果
    h, w = img.shape[:2]
    if img.ndim == 2:
        # 灰度图，转换为3通道以便显示
        img_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        result_vis = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    else:
        img_vis = img.copy()
        result_vis = result.copy()
    
    # 并排拼接
    vis = np.hstack([img_vis, result_vis])
    
    # 添加文字标签
    cv2.putText(vis, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(vis, "Deformed", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 保存结果
    output_path = "deformed/samples/od_pair.png"
    cv2.imwrite(output_path, vis)
    print(f"结果已保存到: {output_path}")
    
    # 也单独保存变形后的图片
    result_path = "deformed/samples/deformed_only.png"
    if result.ndim == 2:
        cv2.imwrite(result_path, result_vis)
    else:
        cv2.imwrite(result_path, result)
    print(f"变形结果已保存到: {result_path}")


if __name__ == "__main__":
    main()
