import os
import time
from typing_extensions import Self

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import resnet18, ResNet18_Weights
# =========================
# HYPER PARAM
# =========================
BATCH_SIZE = 64
EPOCH_NUM = 10

class ResNetClassifier(nn.Module):
    def __init__(self, n_out: int, arch: str = "resnet18",
                 pretrained: bool = True,
                 normalize_imagenet: bool = True):
        """
        n_out: クラス数
        arch: いまは 'resnet18' 前提（必要なら分岐でresnet34/50等に拡張可）
        pretrained: ImageNet学習済み重みを使うか
        normalize_imagenet: 入力をImageNet統計で正規化するか（推奨: True）
        """
        super().__init__()
        self.normalize_imagenet = normalize_imagenet

        # 学習済み重みの取得（オフライン環境なら pretrained=False に）
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = resnet18(weights=weights)

        # 最終fc直前までをbackboneとして使用（conv1..avgpool まで）
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # [B,512,1,1]
        in_feat = base.fc.in_features  # resnet18は512

        # 分類ヘッド
        self.classifier = nn.Linear(in_feat, n_out)

        # ImageNet正規化用パラメータ（学習・推論の両方で使用）
        # register_bufferで学習対象外に固定
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("im_mean", mean)
        self.register_buffer("im_std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W], 値域は [0,1] 推奨（0-255でも動くが正規化の前に /255 を推奨）
        return: [B, n_out]
        """
        if self.normalize_imagenet:
            # 入力が0-255ならここで /255 してから正規化したい場合は下の1行を有効化
            # x = x / 255.0
            x = (x - self.im_mean) / self.im_std

        feat = self.backbone(x)       # [B,512,1,1]
        feat = torch.flatten(feat, 1)  # [B,512]
        logits = self.classifier(feat) # [B,n_out]
        return logits

# =========================
# 学習クラス（時系列なし）
# =========================
class deep_learning:
    def __init__(self, n_out: int = 2):
        # device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("device:", self.device)

        # ★ ここをResNetへ変更 ★
        #   - pretrained=True なら ImageNet重みを使用
        #   - normalize_imagenet=True で内部でImageNet正規化
        self.net = ResNetClassifier(
            n_out=n_out,
            arch="resnet18",
            pretrained=True,            # オフライン環境/重み不要なら False
            normalize_imagenet=True     # 可能なら True 推奨
        ).to(self.device)

        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-4, eps=1e-8, weight_decay=1e-5)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=EPOCH_NUM, eta_min=1e-6)

        balance_weights = torch.tensor([1.0, 2.1], device=self.device, dtype=torch.float32)
        self.criterion = nn.CrossEntropyLoss(weight=balance_weights)

        self.first_flag = True
        self.loss_all = 0.0
        self.results_train = {'loss': [], 'accuracy': []}

        torch.backends.cudnn.benchmark = False
        torch.autograd.set_detect_anomaly(True)

    # -------------------------
    # データ追加
    # img: [H, W, C] (np or torch) / dtypeはfloat32推奨（0-255なら後段で /255 してもOK）
    # label: int クラスID（CrossEntropyLoss 用）
    # -------------------------
    def make_dataset(self, img, label: int):
        """
        画像とラベルを一時リストにappendするだけ
        img: [H,W,C], label: int
        """
        x = torch.tensor(img, dtype=torch.float32)  # [H,W,C]
        if x.dim() != 3 or x.shape[2] != 3:
            raise ValueError("img must be [H,W,3]")

        # 必要なら正規化
        # if x.max() > 1.0:
        #     x = x / 255.0

        x = x.permute(2, 0, 1).unsqueeze(0)  # [1,C,H,W]
        t = torch.tensor([int(label)], dtype=torch.long)  # [1]

        if not hasattr(self, "x_list"):
            self.x_list, self.t_list = [], []

        self.x_list.append(x)
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