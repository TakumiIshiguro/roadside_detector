import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset

PATH = "/home/takumi/catkin_ws/src/roadside_detector/data/dataset/aug/dataset.pt"


def show_tensor_images(path, n_max=50, title=None):
    """
    .pt から画像テンソルや TensorDataset を読み込み、先頭 n_max 枚を可視化する。
    対応入力:
      - torch.Tensor: (N,C,H,W) / (C,H,W) / (H,W,C)
      - TensorDataset: (images, labels) など（images のみ必須）
    値域:
      - uint8(0..255) / float([0,1]) を自動判定して描画
    """
    obj = torch.load(path, map_location="cpu")

    # --- 画像テンソルの抽出 ---
    labels = None
    if isinstance(obj, TensorDataset):
        tensors = obj.tensors
        if len(tensors) == 0:
            raise ValueError("Empty TensorDataset.")
        arr = tensors[0]
        if len(tensors) > 1:
            labels = tensors[1]
    elif isinstance(obj, torch.Tensor):
        arr = obj
    elif isinstance(obj, (tuple, list)) and len(obj) > 0 and isinstance(obj[0], torch.Tensor):
        arr = obj[0]
        if len(obj) > 1 and isinstance(obj[1], torch.Tensor):
            labels = obj[1]
    elif isinstance(obj, dict):
        # よくあるキー名に対応
        for k in ["x_cat", "images", "x", "data"]:
            if k in obj and isinstance(obj[k], torch.Tensor):
                arr = obj[k]
                break
        else:
            raise ValueError("Dict does not contain an image tensor (tried keys: x_cat, images, x, data).")
        for k in ["t_cat", "labels", "y", "targets"]:
            if k in obj and isinstance(obj[k], torch.Tensor):
                labels = obj[k]
                break
    else:
        raise TypeError(f"Unsupported object type loaded from {path}: {type(obj)}")

    # --- 次元を (N,C,H,W) に揃える ---
    if arr.ndim == 3:
        # いずれか：(C,H,W) or (H,W,C)
        if arr.shape[0] in (1, 3):         # (C,H,W)
            arr = arr.unsqueeze(0)         # (1,C,H,W)
        elif arr.shape[-1] in (1, 3):      # (H,W,C)
            arr = arr.permute(2, 0, 1)     # (C,H,W)
            arr = arr.unsqueeze(0)         # (1,C,H,W)
        else:
            raise ValueError(f"Unexpected 3D shape: {tuple(arr.shape)}")
    elif arr.ndim == 4:
        # (N,C,H,W) or (N,H,W,C)
        if arr.shape[1] not in (1, 3) and arr.shape[-1] in (1, 3):
            arr = arr.permute(0, 3, 1, 2)  # (N,H,W,C) → (N,C,H,W)
    else:
        raise ValueError(f"Unsupported tensor ndim: {arr.ndim}")

    N = arr.shape[0]
    n_show = min(N, n_max)

    # --- 値域を判定しつつ描画用に整形 ---
    # uint8 → [0,1] に、float でも [0,1] 外なら per-image min-max で正規化（安全）
    def to_display_img(x_chw: torch.Tensor) -> np.ndarray:
        x = x_chw.detach().cpu()
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        else:
            # 値域チェック（0..1 から外れていたら min-max）
            x_min, x_max = x.amin(), x.amax()
            if (x_min < 0.0) or (x_max > 1.0):
                # per-image min-max 正規化（定数画像は0に）
                imin, imax = x.min(), x.max()
                if float(imax - imin) > 1e-6:
                    x = (x - imin) / (imax - imin)
                else:
                    x = torch.zeros_like(x)

        img = x.numpy()
        img = np.transpose(img, (1, 2, 0))  # (C,H,W)→(H,W,C)
        if img.shape[2] == 1:
            img = img.squeeze(2)  # (H,W)
        return img

    # --- グリッド計算（最大10列） ---
    cols = min(10, n_show)
    rows = math.ceil(n_show / cols)

    fig = plt.figure(figsize=(2.2 * cols, 2.2 * rows))
    if title:
        fig.suptitle(title, fontsize=14)

    for i in range(n_show):
        ax = plt.subplot(rows, cols, i + 1)
        img = to_display_img(arr[i])
        if img.ndim == 2:
            ax.imshow(img, cmap="gray", interpolation="nearest")
        else:
            ax.imshow(img, interpolation="nearest")
        ax.set_axis_off()
        # ラベルがあればサブタイトル表示
        if labels is not None and i < len(labels):
            try:
                lab = int(labels[i].item())
                ax.set_title(str(lab), fontsize=9)
            except Exception:
                pass

    plt.tight_layout(rect=(0, 0, 1, 0.96) if title else None)
    plt.show()


# 実行
show_tensor_images(PATH, n_max=50, title="dataset preview")
