# Research_Project
ML Based Vision systems


# Out-of-Distribution (OOD) Detection Evaluation
## ODIN and ReAct Framework

---

## 1. Introduction

This repository implements and evaluates two Out-of-Distribution (OOD) detection methods:

1. ODIN (Out-of-DIstribution detector for Neural networks)
2. ReAct (Rectified Activations for OOD Detection)

The framework measures how well a trained neural network can distinguish between In-Distribution (ID) and Out-of-Distribution (OOD) samples.

---

## 2. Evaluation Metrics

The following performance metrics are computed:

- AUROC (Area Under the Receiver Operating Characteristic Curve)
- FPR@95TPR (False Positive Rate at 95% True Positive Rate)
- Inference Latency (milliseconds per image)
- Decision Latency (ReAct only)

---

## 3. System Requirements

- Python 3.8 or higher
- PyTorch
- torchvision
- NumPy
- scikit-learn
- tqdm

Install dependencies using:

pip install -r requirements.txt

GPU acceleration is supported if CUDA is available.

---

## 4. Dataset Structure

Both ID and OOD datasets must follow the ImageFolder format:

dataset/
 ├── class1/
 │    ├── image1.png
 │    ├── image2.png
 ├── class2/
 │    ├── image1.png
 │    ├── image2.png

Separate folders must be prepared for:

- In-Distribution dataset
- Out-of-Distribution dataset

---

## 5. ODIN Evaluation

### 5.1 Configuration

Open the ODIN script and set the following paths:

MODEL_PATH = "/path/to/models/densenet100.pth"

ID_DATA_PATH = "/path/to/id_dataset/"

OOD_DATA_PATH = "/path/to/ood_dataset/"

---

### 5.2 Execution

Run the script using:

python odin_script.py

### 5.3 Method Description

ODIN improves OOD detection by:

1. Applying temperature scaling to logits
2. Adding small adversarial perturbations to inputs
3. Using maximum softmax probability as detection score

---

## 6. ReAct Evaluation

### 6.1 Configuration

Open the ReAct script and set:

ID_DATASET_PATH = "/path/to/id_dataset/"

OOD_DATASET_PATH = "/path/to/ood_dataset/"

MODEL_CHECKPOINT_PATH = "/path/to/models/resnet50.pth"

---

### 6.2 Execution

Run the script using:

python react_script.py


---

### 6.3 Method Description

ReAct enhances OOD detection by:

1. Extracting intermediate feature representations
2. Clipping high-magnitude activations based on a percentile threshold
3. Computing an energy-based score:

Energy(x) = logsumexp(logits)

Higher energy values typically correspond to In-Distribution samples.

---


## 8. Hardware Support

- CPU supported
- CUDA GPU supported
- Tested on NVIDIA GPUs and Jetson platforms

The device is selected automatically:

device = "cuda" if torch.cuda.is_available() else "cpu"

---

## 9. Notes

- Ensure that the model checkpoint matches the architecture used.
- ID dataset should match the model’s training distribution.
- Image size requirements:
  - ODIN: 32 × 32 images
  - ReAct: 224 × 224 images

---


