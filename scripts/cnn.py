import os
import time
from typing_extensions import Self

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torchvision.transforms.v2 as Tv2

from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import (
    mobilenet_v3_large, MobileNet_V3_Large_Weights,
    resnet18, ResNet18_Weights,
    resnet50, ResNet50_Weights,
    vit_b_16, ViT_B_16_Weights,
)

BATCH_SIZE = 64
EPOCH_NUM = 10

class BackboneWrapper(nn.Module):
    def __init__(self, backbone: nn.Module, out_dim: int, n_out: int):
        """
        backbone: 特徴抽出部 (nn.Module)
        out_dim : 特徴ベクトル次元数
        n_out   : 出力クラス数
        """
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(out_dim, n_out)

        # ImageNet 正規化
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("im_mean", mean)
        self.register_buffer("im_std", std)

    def forward(self, x):
        # 入力は [0,1] 前提（0-255 の場合は事前に /255.0 してください）
        x = (x - self.im_mean) / self.im_std

        feat = self.backbone(x)               # [B,out_dim] or [B,out_dim,1,1]
        logits = self.classifier(feat)        # [B,n_out]
        return logits

from torchvision.models import (
    mobilenet_v3_large, MobileNet_V3_Large_Weights,
    resnet18, ResNet18_Weights,
    resnet50, ResNet50_Weights,
    vit_b_16, ViT_B_16_Weights,
)

def build_model(name: str, n_out: int = 2) -> nn.Module:
    """
    利用可:
      "mobilenet_v3_large", "resnet18", "resnet50", "vit_b16"
    いずれも最終分類層を Identity に置換し、直前の特徴ベクトルを BackboneWrapper に渡す
    """
    key = name.lower()

    if key in ("mobilenet_v3_large", "mobilenetv3large", "mnetv3l"):
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
        base = mobilenet_v3_large(weights=weights)
        # classifier: [Linear(960→1280), Hardswish, Dropout, Linear(1280→1000)]
        base.classifier[-1] = nn.Identity()        # 最終Linearだけ外す
        out_dim = base.classifier[0].out_features  # 1280
        backbone = base                            # 出力: [B,1280]
        return BackboneWrapper(backbone, out_dim, n_out)

    elif key == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1
        base = resnet18(weights=weights)
        out_dim = base.fc.in_features              # 512
        base.fc = nn.Identity()                    # 最終Linearだけ外す
        backbone = base                            # 出力: [B,512]
        return BackboneWrapper(backbone, out_dim, n_out)

    elif key == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V1
        base = resnet50(weights=weights)
        out_dim = base.fc.in_features              # 2048
        base.fc = nn.Identity()                    # 最終Linearだけ外す
        backbone = base                            # 出力: [B,2048]
        return BackboneWrapper(backbone, out_dim, n_out)

    elif key in ("vit_b16", "vit-b16", "vit_b_16"):
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        base = vit_b_16(weights=weights)
        out_dim = base.heads.head.in_features      # 768
        base.heads.head = nn.Identity()            # 最終Linearだけ外す (LayerNormは残る)
        backbone = base                            # 出力: [B,768]
        return BackboneWrapper(backbone, out_dim, n_out)

    else:
        raise ValueError(f"Unknown model name: {name}")
 
class deep_learning:
    def __init__(self, n_out: int = 2):
        # device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("device:", self.device)

        backbone = "mobilenet_v3_large"
        # backbone = "resnet18"
        # backbone = "resnet50"
        # backbone = "vit_b16"
        self.net = build_model(backbone, n_out=n_out).to(self.device)

        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-4, eps=1e-8, weight_decay=1e-5)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=EPOCH_NUM, eta_min=1e-6)

        balance_weights = torch.tensor([1.0, 2.1], device=self.device, dtype=torch.float32)
        self.criterion = nn.CrossEntropyLoss(weight=balance_weights)

        self.first_flag = True
        self.loss_all = 0.0
        self.results_train = {'loss': [], 'accuracy': []}

        torch.backends.cudnn.benchmark = False
        torch.autograd.set_detect_anomaly(True)

        self.aug_trans = Tv2.Compose([
            Tv2.ColorJitter(brightness=0.2, contrast=0.2,
                            saturation=0.2, hue=0.02),
            Tv2.RandomHorizontalFlip(p=0.5),
            Tv2.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
            Tv2.RandomErasing(p=0.2, scale=(0.02, 0.08), ratio=(0.3, 3.3))
        ])

    def make_dataset(self, img, label: int):
        """
        画像とラベルを一時リストにappendするだけ
        img: [H,W,C], label: int
        """
        x = torch.tensor(img, dtype=torch.float32)  # [H,W,C]
        if x.dim() != 3 or x.shape[2] != 3:
            raise ValueError("img must be [H,W,3]")

        # [H,W,C] → [C,H,W]
        x = x.permute(2, 0, 1)

        # 値域が0-255なら正規化
        # if x.max() > 1.0:
        #     print("0-255→0-1")
        #     x = x / 255.0  # v2は0〜1 Tensor対応

        # augmentation 適用
        x_aug = self.aug_trans(x)   # [C,H,W]

        # [1,C,H,W] に拡張
        x_aug = x_aug.unsqueeze(0)

        # ラベル
        t = torch.tensor([int(label)], dtype=torch.long)  # [1]

        if not hasattr(self, "x_list"):
            self.x_list, self.t_list = [], []

        self.x_list.append(x_aug)
        self.t_list.append(t)

        print(f"total {len(self.x_list)} samples")
        return len(self.x_list)

    def finalize_dataset(self):
        """
        appendされたデータをまとめてcatしてTensorに変換
        """
        if not hasattr(self, "x_list") or len(self.x_list) == 0:
            raise RuntimeError("No data appended. Call make_dataset() first.")

        self.x_cat = torch.cat(self.x_list, dim=0)  # [N,C,H,W]
        self.t_cat = torch.cat(self.t_list, dim=0)  # [N]
        self.first_flag = False

        # メモリ解放
        del self.x_list
        del self.t_list

        print("Final dataset shapes -> X:", self.x_cat.shape, "Y:", self.t_cat.shape)
        return self.x_cat, self.t_cat

    # -------------------------
    # 学習
    # -------------------------
    def training(self):
        if not hasattr(self, 'x_cat') or not hasattr(self, 't_cat'):
            raise RuntimeError("No dataset yet. Call make_dataset() first.")

        dataset = TensorDataset(self.x_cat, self.t_cat)  # X:[N,C,H,W], Y:[N]
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)

        self.net.train()
        final_acc, final_loss = 0.0, 0.0

        for epoch in range(EPOCH_NUM):
            epoch_loss = 0.0
            correct = 0
            total = 0

            for x_train, y_train in train_loader:
                x_train = x_train.to(self.device, non_blocking=True)  # [B,C,H,W]
                y_train = y_train.to(self.device, non_blocking=True)  # [B]

                self.optimizer.zero_grad()
                logits = self.net(x_train)                # [B,n_out]
                loss = self.criterion(logits, y_train)

                loss.backward()
                self.optimizer.step()

                # ログ
                epoch_loss += loss.item() * x_train.size(0)
                preds = torch.argmax(logits, dim=1)       # [B]
                correct += (preds == y_train).sum().item()
                total += y_train.size(0)

            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()

            avg_loss = epoch_loss / max(total, 1)
            acc = correct / max(total, 1)
            print(f'epoch [{epoch+1}/{EPOCH_NUM}] loss: {avg_loss:.4f} acc: {acc:.4f} lr: {current_lr:.6f}')

            self.results_train['loss'].append(avg_loss)
            self.results_train['accuracy'].append(acc)

            final_acc, final_loss = acc, avg_loss

        print("Finish learning")
        return final_acc, final_loss

    # -------------------------
    # 推論（単一画像）
    # img: [H,W,C]
    # returns: (pred_class:int, confidence:float, probs:Tensor[n_out])
    # -------------------------
    def test(self, img):
        self.net.eval()
        with torch.no_grad():
            x = torch.tensor(img, dtype=torch.float32)
            # x = x / 255.0  # 必要なら有効化
            x = x.permute(2, 0, 1).unsqueeze(0).to(self.device)  # [1,C,H,W]
            logits = self.net(x)                 # [1,n_out]
            probs = F.softmax(logits, dim=1)[0]  # [n_out]
            conf, pred = torch.max(probs, dim=0)
            # print("softmax:", probs.detach().cpu())
            # print("confidence:", conf.item(), "predicted:", pred.item())
            return int(pred.item()), float(conf.item()), probs.detach().cpu()

    def save_dataset(self, save_root, file_name):
        path = save_root + time.strftime("%Y%m%d_%H:%M:%S")
        os.makedirs(path, exist_ok=True)
        dataset = TensorDataset(self.x_cat, self.t_cat)
        torch.save(dataset, os.path.join(path, file_name))
        print("save_dataset_tensor")

    def save(self, save_root):
        path = save_root + time.strftime("%Y%m%d_%H:%M:%S")
        os.makedirs(path, exist_ok=True)
        torch.save(self.net.state_dict(), os.path.join(path, 'model.pt'))
        print("saved to:", path)

    def load_dataset(self, path):
        dataset = torch.load(path)  # TensorDataset
        self.x_cat, self.t_cat = dataset.tensors
        print("loaded_dataset:", path)
        print("shapes -> X:", self.x_cat.shape, "Y:", self.t_cat.shape)
        
    def load(self, load_path):
        self.net.load_state_dict(torch.load(load_path, map_location=self.device))
        self.net.to(self.device)
        print("Loaded model from:", load_path)

if __name__ == '__main__':
    dl = deep_learning()