
from __future__ import print_function

import argparse
import math
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score


sys.modules["densenet"] = sys.modules[__name__]

# =========================================================
# DenseNet Architecture Components
# =========================================================

class BasicBlock(nn.Module):
    

    def __init__(self, in_planes: int, out_planes: int):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, 1, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.relu(self.bn1(x)))
        return torch.cat([x, out], 1)


class BottleneckBlock(nn.Module):

    def __init__(self, in_planes: int, out_planes: int):
        super().__init__()
        inter_planes = out_planes * 4

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, 1, 1, 0, bias=False)

        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, 3, 1, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        return torch.cat([x, out], 1)


class TransitionBlock(nn.Module):

    def __init__(self, in_planes: int, out_planes: int):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, 1, 1, 0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.relu(self.bn1(x)))
        return F.avg_pool2d(out, 2)


class DenseBlock(nn.Module):

    def __init__(self, nb_layers: int, in_planes: int, growth_rate: int, block):
        super().__init__()
        layers = [block(in_planes + i * growth_rate, growth_rate) for i in range(int(nb_layers))]
        self.layer = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class DenseNet3(nn.Module):

    def __init__(self, depth: int = 100, num_classes: int = 100, growth_rate: int = 12, reduction: float = 0.5):
        super().__init__()

        in_planes = 2 * growth_rate
        n = (depth - 4) / 3
        n = n / 2  # bottleneck version
        block = BottleneckBlock

        # Initial convolution
        self.conv1 = nn.Conv2d(3, in_planes, 3, 1, 1, bias=False)

        # Dense Block 1
        self.block1 = DenseBlock(n, in_planes, growth_rate, block)
        in_planes = int(in_planes + n * growth_rate)
        self.trans1 = TransitionBlock(in_planes, int(math.floor(in_planes * reduction)))
        in_planes = int(math.floor(in_planes * reduction))

        # Dense Block 2
        self.block2 = DenseBlock(n, in_planes, growth_rate, block)
        in_planes = int(in_planes + n * growth_rate)
        self.trans2 = TransitionBlock(in_planes, int(math.floor(in_planes * reduction)))
        in_planes = int(math.floor(in_planes * reduction))

        # Dense Block 3
        self.block3 = DenseBlock(n, in_planes, growth_rate, block)
        in_planes = int(in_planes + n * growth_rate)

        # Classifier
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_planes, num_classes)
        self.in_planes = in_planes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.in_planes)
        return self.fc(out)


# Allow loading older checkpoints safely
torch.serialization.add_safe_globals([DenseNet3])

# =========================================================
# Dataset Configuration
# =========================================================
MODEL_PATH = "/home/mj/rp/testing/Jetson/odin_jetson/models/densenet100.pth"
ID_DATA_PATH = "/home/mj/cifar100_png/test/"
OOD_DATA_PATH = "/home/mj/rp/testing/Jetson/ood/LSUN/"

MEAN = (125.3 / 255, 123.0 / 255, 113.9 / 255)
STD = (63.0 / 255, 62.1 / 255, 66.7 / 255)

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# =========================================================
# ODIN Scoring
# =========================================================

def odin_score(net: nn.Module, x: torch.Tensor, eps: float, T: float) -> np.ndarray:

    x.requires_grad_(True)

    # Forward pass with temperature scaling
    outputs = net(x) / T
    pred = outputs.detach().max(1)[1]
    loss = F.cross_entropy(outputs, pred)

    # Backprop for input gradient
    net.zero_grad(set_to_none=True)
    loss.backward()

    gradient = torch.sign(x.grad)

    # Normalize gradient by channel std
    gradient[:, 0] /= (63.0 / 255.0)
    gradient[:, 1] /= (62.1 / 255.0)
    gradient[:, 2] /= (66.7 / 255.0)

    # Input perturbation
    perturbed = x.detach() - eps * gradient
    perturbed = perturbed.clamp(-5, 5)

    # Re‑evaluate model
    with torch.no_grad():
        outputs = net(perturbed) / T
        score = torch.softmax(outputs, dim=1).max(1)[0]

    return score.detach().cpu().numpy()


def collect(net, loader, eps, T, device, name):

    scores = []
    total_samples = len(loader.dataset)
    processed = 0

    print(f"\n>>> Collecting {name} ({total_samples} samples)")

    for i, (x, _) in enumerate(loader):
        batch_size = x.size(0)
        processed += batch_size

        if i % 50 == 0:
            print(f"{name}: {processed}/{total_samples}")

        x = x.to(device)
        scores.extend(odin_score(net, x, eps, T))

    return np.array(scores)

# =========================================================
# Evaluation Metrics
# =========================================================

def auroc(in_s, out_s):
    labels = np.concatenate([np.ones(len(in_s)), np.zeros(len(out_s))])
    scores = np.concatenate([in_s, out_s])
    return roc_auc_score(labels, scores)


def fpr95(in_s, out_s):
    threshold = np.percentile(in_s, 5)
    return np.mean(out_s >= threshold)

# =========================================================
# Main Execution
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="ODIN OOD Evaluator")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--magnitude", type=float, default=0.002)
    parser.add_argument("--temperature", type=float, default=1000)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print("Loading model...")
    net = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    net = net.to(device).eval()

    BATCH = 64

    id_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(ID_DATA_PATH, transform),
        batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)

    ood_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(OOD_DATA_PATH, transform),
        batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)

    # ---- latency measurement ----
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    id_scores = collect(net, id_loader, args.magnitude, args.temperature, device, "ID")
    ood_scores = collect(net, ood_loader, args.magnitude, args.temperature, device, "OOD")

    if device.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.perf_counter()

    total_images = len(id_scores) + len(ood_scores)
    latency_ms = (end_time - start_time) / total_images * 1000

    # ---- metrics ----
    au = auroc(id_scores, ood_scores) * 100
    fp = fpr95(id_scores, ood_scores) * 100

    print("\nRESULTS")
    print(f"AUROC  : {au:.2f}")
    print(f"FPR95  : {fp:.2f}")
    print(f"Latency: {latency_ms:.3f} ms / image")


if __name__ == "__main__":
    main()
