
# AIGM-Cap3D: Assessing and Improving the Generalizability of Text-to-3D Models using Cap3D

**Course:**  CMPT 420/728 Deep Learning 
**Authors:** Archita Srivastava, Mohammad Taghizadeh, Sydney Widjaja 
**Institution:** Simon Fraser University  

## Overview

Text-to-3D generation remains one of the most challenging frontiers in generative AI. Key obstacles include
the scarcity of high-quality text-3D paired data, reliance on synthetic data and 2D image-text knowledge
to infer 3D structure, and the **Janus problem** — where models lack global geometric coherence and produce
view-inconsistent shapes.

This project investigates these limitations by fine-tuning
[Shap-E](https://github.com/openai/shap-e), OpenAI's text-to-3D diffusion model, on the
[Cap3D](https://github.com/crockwell/Cap3D) dataset — a large-scale collection of 3D objects from
Objaverse paired with detailed natural language captions.

### Research Questions

1. How do current text-to-3D architectures struggle with semantic-geometric alignment under zero-shot conditions?
2. Does training on more detailed natural language descriptions help a model reliably understand complex prompts?

---

## Models Considered

Three publicly available text-to-3D models were evaluated as part of this study:

| Model | Source |
|---|---|
| **Point-E** | OpenAI / Hugging Face |
| **Shap-E** | OpenAI / Hugging Face |
| **Fantasia3D** | Hugging Face |

---

## Dataset

**Cap3D** was chosen for its large-scale captioned 3D data, enabling an in-depth analysis of how
richer natural language descriptions affect text-to-3D semantic alignment. It provides per-object
captions generated from multi-view renders of Objaverse assets using BLIP-2 and GPT-4.

---


## Setup & Installation

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

## Methodology

### Zero-Shot Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| Guidance Scale | 17.5 | Forces strict adherence to the text prompt |
| Num Steps | 64 | Balances generation speed with geometric detail |
| Sigma Range | 1e-3 to 160 | Defines the diffusion noise schedule for shape resolution |
| Evaluation Set | N = 100 | Results averaged over 100 diverse prompts for statistical significance |

### Fine-Tuning Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| Learning Rate | 1 × 10⁻⁵ | Slow convergence to prevent catastrophic forgetting of 3D priors |
| Batch Size | 8 | Optimized for VRAM efficiency on transformer blocks |
| Max Epochs | 15 | Sufficient depth for the model to see the full Cap3D subset |
| Train/Val Split | 80/20 | Standard split to monitor generalization |

---

## Results

### Quantitative Evaluation

| Metric | Zero-Shot | Fine-Tuned | Delta | Category |
|---|---|---|---|---|
| Chamfer Distance ↓ | 0.0334 | 0.0335 | +0.0001 | Geometry |
| F-Score ↑ | 0.0796 | 0.0827 | +0.0031 | Geometry |
| CLIP Score ↑ | 0.2519 | 0.2509 | -0.0010 | Semantic |
| CLIP Similarity ↑ | 0.8106 | 0.8137 | +0.0031 | Semantic |
| R-Precision ↑ | 0.1600 | 0.1600 | +0.0000 | Semantic |
| LPIPS ↓ | 0.2624 | 0.2633 | +0.0009 | Perceptual |

Fine-tuning yields modest improvements in **geometric accuracy** (F-Score) and **semantic similarity**
(CLIP Similarity), while zero-shot performs marginally better on raw CLIP Score and perceptual realism.
Results suggest fine-tuning on Cap3D captions helps with structural fidelity but does not dramatically
shift semantic alignment at this scale.

Detailed results: `eval_results_zeroshot.csv` · `eval_results_finetuned.csv`

### Qualitative Comparison

Below is a qualitative comparison of Cap3D ground truth meshes, zero-shot generations, and fine-tuned
generations conditioned on the original Cap3D caption.

![Qualitative Comparison](src/shap-e-zeroshot/predicted_previews/comparison_sample.png)

> *Left to right: Cap3D Ground Truth · Shap-E Zero-Shot · Shap-E Fine-Tuned*

See [`comparison.ipynb`](src/shap-e-zeroshot/comparison.ipynb) for the full visual analysis.

---

## Directory Structure


```AIGM-Cap3D/
├── src/
│   └── shap-e-zeroshot/
│       ├── generated/                    # Zero-shot generated meshes
│       ├── graphs/                       # Training loss curves
│       ├── latents/                      # Encoded shape latents for training
│       ├── predicted_previews/           # Zero-shot preview renders
│       ├── predicted_previews_finetuned/ # Fine-tuned preview renders
│       ├── predicted_previews_guidance/  # Guidance-based Zero-shot preview renders
│       ├── previews_gif/                 # Ground-Truth preview renders
        ├── scripts/                              # All runnable scripts
        │   ├── data.py                           # Dataset preparation
        │   ├── eval.py                           # Evaluation metrics
        │   ├── fine-tune.py                      # Fine-tuning pipeline
        │   ├── inference-post-train.py           # Inference with fine-tuned model
        │   ├── inference-zero-shot.py            # Zero-shot baseline inference
        │   └── view.py                           # Render / visualize generated shapes
│       ├── shap-e/                       # Shap-E submodule
│       ├── shap_e_model_cache/           # Cached model weights (not tracked)
│       ├── comparison.ipynb              # Visual comparison notebook
│       ├── eval_results_finetuned.csv    # Fine-tuned evaluation results
│       └── eval_results_zeroshot.csv     # Zero-shot evaluation results
├── cap3d_splits.py                       # Dataset split generation
├── downloaded_objects_split.json         # Object split index (not tracked)
├── requirements.txt
└── .gitignore
```















## Shap-E setup

```bash
cd src/shap-e-zeroshot/shap-e
pip install -e .
```

### How to Run

All scripts are located in `src/shap-e-zeroshot/scripts/`.

1. Prepare data splits
```bash
python cap3d_splits.py
```

2. Zero-shot generation
```bash
python scripts/zero-shot.py
```

3. Fine-tune the model
```bash
python scripts/fine-tune.py
```

4. Run inference with fine-tuned model
```bash
python scripts/inference-post-train.py
```

### 5. Evaluate results
```bash
python scripts/eval.py
```

