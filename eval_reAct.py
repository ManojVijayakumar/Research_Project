# ============================================================
# USER CONFIG
# ============================================================

ID_DATASET_PATH = "/home/mj/rp/testing/Jetson/react_jetson/id/val"
OOD_DATASET_PATH = "/home/mj/rp/testing/Jetson/ood/LSUN/"
MODEL_CHECKPOINT_PATH = "/home/mj/rp/testing/Jetson/react_jetson/models/resnet50_imagenet.pth"

BATCH_SIZE = 25
REACT_PERCENTILE = 90

# ============================================================

import time
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True

# -------------------------
#FPR@95TPR
# -------------------------
def fpr_at_95_tpr(id_scores, ood_scores):
    id_scores = np.sort(id_scores)
    thresh = id_scores[int(0.05 * len(id_scores))]
    return np.mean(ood_scores >= thresh)

# -------------------------
# Dataset
# -------------------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])

print("Loading datasets...")

id_dataset = datasets.ImageFolder(ID_DATASET_PATH, transform=transform)
ood_dataset = datasets.ImageFolder(OOD_DATASET_PATH, transform=transform)

id_loader = torch.utils.data.DataLoader(
    id_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
)

ood_loader = torch.utils.data.DataLoader(
    ood_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
)

# -------------------------
# Model
# -------------------------
print("Loading ResNet-50...")

model = resnet50(weights=None)


ckpt = torch.load(MODEL_CHECKPOINT_PATH, map_location="cpu")

if "state_dict" in ckpt:
    ckpt = ckpt["state_dict"]
if "model" in ckpt:
    ckpt = ckpt["model"]

new_state = {}
for k, v in ckpt.items():
    k = k.replace("module.", "")
    new_state[k] = v

model.load_state_dict(new_state, strict=False)
model = model.to(device)
model.eval()



def extract_features(model, x):
    features = []

    def hook(module, inp, out):
        features.append(out)

    handle = model.avgpool.register_forward_hook(hook)
    _ = model(x)
    handle.remove()

    feat = torch.flatten(features[0], 1)
    return feat



print("\nEstimating ReAct threshold...")

activation_log = []

with torch.no_grad():
    for images, _ in tqdm(id_loader):
        images = images.to(device)
        feat = extract_features(model, images)
        activation_log.append(feat.cpu())

activation_log = torch.cat(activation_log, dim=0)
react_threshold = np.percentile(
    activation_log.numpy(), REACT_PERCENTILE
)

print(f"ReAct threshold (p={REACT_PERCENTILE}) = {react_threshold:.4f}")


def get_energy_scores(loader, name, measure_time=False):

    scores = []
    total_time = 0.0
    total_images = 0

    with torch.no_grad():
        for images, _ in tqdm(loader, desc=name):

            images = images.to(device)
            batch_size = images.size(0)

            # ---------------- TIME START ----------------
            if measure_time:
                torch.cuda.synchronize() if device == "cuda" else None
                start = time.perf_counter()
            # ------------------------------------------------

            feat = extract_features(model, images)

            # ReAct clipping
            feat = torch.clamp(feat, max=react_threshold)

            logits = model.fc(feat)

            # Energy score
            energy = torch.logsumexp(logits, dim=1)

            # ---------------- TIME END ----------------
            if measure_time:
                torch.cuda.synchronize() if device == "cuda" else None
                end = time.perf_counter()
                total_time += (end - start)
                total_images += batch_size
            # ------------------------------------------------

            scores.append(energy.cpu())

    if measure_time:
        return torch.cat(scores).numpy(), total_time, total_images
    else:
        return torch.cat(scores).numpy()



print("\nRunning ID inference...")
id_scores, id_time, id_count = get_energy_scores(id_loader, "ID", measure_time=True)

print("\nRunning OOD inference...")
ood_scores, ood_time, ood_count = get_energy_scores(ood_loader, "OOD", measure_time=True)

full_time = id_time + ood_time
full_images = id_count + ood_count

ms_per_image = (full_time / full_images) * 1000

# -------------------------
# Metrics
# -------------------------
labels = np.concatenate([
    np.ones_like(id_scores),
    np.zeros_like(ood_scores),
])
scores = np.concatenate([id_scores, ood_scores])

auroc = roc_auc_score(labels, scores)
fpr95 = fpr_at_95_tpr(id_scores, ood_scores)



print("\nMeasuring decision latency...")

sorted_id = np.sort(id_scores)
decision_threshold = sorted_id[int(0.05 * len(sorted_id))]

all_scores = np.concatenate([id_scores, ood_scores])

repeat = 2000

start_time = time.perf_counter()

for _ in range(repeat):
    predictions = (all_scores >= decision_threshold).astype(np.int32)

end_time = time.perf_counter()

total_time = end_time - start_time

ns_per_image = (total_time / (len(all_scores) * repeat)) * 1e9


print("\n========== ReAct OOD Evaluation ==========")
print(f"ID samples         : {len(id_scores)}")
print(f"OOD samples        : {len(ood_scores)}")
print(f"AUROC              : {auroc * 100:.2f} %")
print(f"FPR@95TPR          : {fpr95 * 100:.2f} %")
print(f"Full latency       : {ms_per_image:.3f} ms / image")
print(f"Decision latency   : {ns_per_image:.2f} ns / image")
print("===========================================")

