# AIGM-Cap3D: Assessing and Improving the Generalizability of Text-to-3D Models using Cap3D

**Course:** CMPT 420/728 Deep Learning  
**Authors:** Archita Srivastava, Mohammad Taghizadeh, Sydney Widjaja  
**Institution:** Simon Fraser University  

---

## Overview

Text-to-3D generation remains one of the most challenging frontiers in generative AI. Key obstacles include
the scarcity of high-quality text-3D paired data, reliance on synthetic data and 2D image-text knowledge
to infer 3D structure, and the **Janus problem** - where models lack global geometric coherence and produce
view-inconsistent shapes.

This project investigates these limitations by evaluating and fine-tuning three publicly available
text-to-3D models on the [Cap3D](https://github.com/crockwell/Cap3D) dataset — a large-scale collection
of 3D objects from Objaverse paired with detailed natural language captions.

### Research Questions

1. How do current text-to-3D architectures struggle with semantic-geometric alignment under zero-shot conditions?
2. Does training on more detailed natural language descriptions help a model reliably understand complex prompts?

---

## Models

Three publicly available text-to-3D models were evaluated as part of this study:

| Model | Source | Fine-Tuned |
|---|---|---|
| **Shap-E** | OpenAI / Hugging Face | Yes |
| **Point-E** | OpenAI / Hugging Face | Yes |
| **Fantasia3D** | Hugging Face | Zero-shot only |

---

## Dataset

**Cap3D** was chosen for its large-scale captioned 3D data, enabling an in-depth analysis of how
richer natural language descriptions affect text-to-3D semantic alignment. It provides per-object
captions generated from multi-view renders of Objaverse assets using BLIP-2 and GPT-4.

---

## Setup & Installation

> The following setup instructions apply to the **Shap-E** pipeline. Point-E and Fantasia3D were evaluated separately using their respective repositories.

### Prerequisites
- Python 3.12.3
- CUDA-compatible GPU (recommended)
- [Blender 3.3.1](https://www.blender.org/download/releases/3-3/) (for rendering previews)

### Install dependencies

```bash
git clone https://github.com/Archita93/AIGM-Cap3D.git
cd AIGM-Cap3D
pip install -r requirements.txt
```

### Shap-E setup

```bash
cd src/shap-e-zeroshot/shap-e
pip install -e .
```

---

## How to Run (Shap-E)

All scripts are located in `src/shap-e-zeroshot/`.

### 1. Prepare the dataset
```bash
python data.py
```

### 2. Zero-shot inference (baseline)
```bash
python inference-zero-shot.py
```

### 3. Fine-tune the model
```bash
python fine-tune.py
```

### 4. Run inference with the fine-tuned model
```bash
python inference-post-train.py
```

### 5. Evaluate results
```bash
python eval.py
```

### 6. Render / view generated shapes
```bash
python view.py
```

---

## Methodology

### Shap-E Zero-Shot Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| Guidance Scale | 17.5 | Forces strict adherence to the text prompt |
| Num Steps | 64 | Balances generation speed with geometric detail |
| Sigma Range | 1e-3 to 160 | Defines the diffusion noise schedule for shape resolution |
| Evaluation Set | N = 100 | Results averaged over 100 diverse prompts for statistical significance |

### Shap-E Fine-Tuning Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| Learning Rate | 1 × 10⁻⁵ | Slow convergence to prevent catastrophic forgetting of 3D priors |
| Batch Size | 8 | Optimized for VRAM efficiency on transformer blocks |
| Max Epochs | 15 | Sufficient depth for the model to see the full Cap3D subset |
| Train/Val Split | 80/20 | Standard split to monitor generalization |

---

## Results

### Quantitative Evaluation

#### Shap-E

| Metric | Zero-Shot | Fine-Tuned | Delta | Category |
|---|---|---|---|---|
| Chamfer Distance ↓ | 0.0334 | 0.0335 | +0.0001 | Geometry |
| F-Score ↑ | 0.0796 | 0.0827 | +0.0031 | Geometry |
| CLIP Score ↑ | 0.2519 | 0.2509 | -0.0010 | Semantic |
| CLIP Similarity ↑ | 0.8106 | 0.8137 | +0.0031 | Semantic |
| R-Precision ↑ | 0.1600 | 0.1600 | +0.0000 | Semantic |
| LPIPS ↓ | 0.2624 | 0.2633 | +0.0009 | Perceptual |

#### Point-E

| Metric | Zero-Shot | Fine-Tuned | Delta | Category |
|---|---|---|---|---|
| Chamfer Distance ↓ | 0.0079 | 0.0071 | -0.0008 | Geometry |
| CLIP Score ↑ | 0.2650 | 0.2970 | +0.0320 | Semantic |
| CLIP Similarity ↑ | 0.9550 | 0.9560 | +0.0010 | Semantic |

#### Fantasia3D (Zero-Shot Only)

| Metric | Zero-Shot | Category |
|---|---|---|
| Chamfer Distance ↓ | 0.4912 | Geometry |
| F-Score ↑ | 0.0046 | Geometry |
| CLIP Score ↑ | 0.1902 | Semantic |
| CLIP Similarity ↑ | 0.7450 | Semantic |
| LPIPS ↓ | 0.3206 | Perceptual |

Fine-tuning on Cap3D captions shows the clearest gains for **Point-E**, with a notable improvement
in CLIP Score (+0.032) and a reduction in Chamfer Distance. **Shap-E** sees modest geometric
improvements (F-Score) but marginal semantic change. **Fantasia3D** was evaluated zero-shot only
and underperforms on geometric metrics, reflecting its sensitivity to optimization stability.

Detailed Shap-E results: `src/shap-e-zeroshot/eval_results_zeroshot.csv` · `src/shap-e-zeroshot/eval_results_finetuned.csv`

### Qualitative Comparison (Shap-E)

Below is a qualitative comparison of Cap3D ground truth meshes, zero-shot generations, and fine-tuned
generations conditioned on the original Cap3D caption.

![Qualitative Comparison](src/shap-e-zeroshot/predicted_previews/comparison_sample.png)

> *Left to right: Cap3D Ground Truth · Shap-E Zero-Shot · Shap-E Fine-Tuned*

See [`comparison.ipynb`](src/shap-e-zeroshot/comparison.ipynb) for the full visual analysis.

---

## Directory Structure

```
AIGM-Cap3D/
├── src/
│   └── shap-e-zeroshot/
│       ├── generated/                    # Zero-shot generated meshes
│       ├── graphs/                       # Training loss curves
│       ├── latents/                      # Encoded shape latents for training
│       ├── predicted_previews/           # Zero-shot preview renders
│       ├── predicted_previews_finetuned/ # Fine-tuned preview renders
│       ├── predicted_previews_guidance/  # Guidance-based zero-shot preview renders
│       ├── previews_gif/                 # Ground truth preview renders
│       ├── scripts/
│       │   ├── data.py                   # Dataset preparation
│       │   ├── eval.py                   # Evaluation metrics
│       │   ├── fine-tune.py              # Fine-tuning pipeline
│       │   ├── inference-post-train.py   # Inference with fine-tuned model
│       │   ├── inference-zero-shot.py    # Zero-shot baseline inference
│       │   └── view.py                   # Render / visualize generated shapes
│       ├── shap-e/                       # Shap-E submodule (OpenAI)
│       ├── shap_e_model_cache/           # Cached model weights (not tracked)
│       ├── comparison.ipynb              # Visual comparison notebook
│       ├── eval_results_finetuned.csv    # Fine-tuned evaluation results
│       └── eval_results_zeroshot.csv     # Zero-shot evaluation results
├── cap3d_splits.py                       # Dataset split generation
├── downloaded_objects_split.json         # Object split index (not tracked)
├── requirements.txt
└── .gitignore
```

---

## References

- Heewoo Jun and Alex Nichol. *Shap-E: Generating Conditional 3D Implicit Functions.* 2023.
  [arXiv:2305.02463](https://arxiv.org/abs/2305.02463)
- Tiange Luo et al. *Cap3D: Scalable 3D Captioning with Pretrained Models.* 2023.
  [arXiv:2306.07279](https://arxiv.org/abs/2306.07279)
- Alex Nichol et al. *Point-E: A System for Generating 3D Point Clouds from Complex Prompts.* 2022.
  [arXiv:2212.08751](https://arxiv.org/abs/2212.08751)
- Rui Chen et al. *Fantasia3D: Disentangling Geometry and Appearance for High-quality Text-to-3D
  Content Creation.* 2023. [arXiv:2303.13873](https://arxiv.org/abs/2303.13873)
- Objaverse: [https://objaverse.allenai.org](https://objaverse.allenai.org)
