# Open Pixel2Play (P2P)

**Open Pixel2Play (P2P)** is an open foundation model trained to play video games in real time. The model takes **visual input (images) and text instructions** and outputs **keyboard and mouse actions**, enabling direct interaction with real game environments.

P2P is trained on **8,000+ hours of human-annotated gameplay videos**. We are actively working on releasing the full dataset. In the meantime, a **toy sample dataset** is available on Hugging Face:  
👉 https://huggingface.co/datasets/guaguaa/p2p-toy-examples

Our smallest model (**150M parameters**) can be trained on **8× H100 GPUs in ~70 hours**.

This repository contains:
- The full **training pipeline** for P2P models  
- **Inference code** for running trained models  
- Integration with **Recap** for real-time interaction with commercial games  

---

## Repository Overview

This repo provides everything needed to:
- Train P2P models from scratch
- Run offline validation
- Serve models for real-time inference
- Connect models to real games via the **Recap** system

To interact with real game environments, you must also install the **Recap** repository:

🔗 https://github.com/elefant-ai/recap

Recap runs on **Windows**, while the inference server runs on **Linux or WSL**.

---

## Installation

### Prerequisites

- **Game environments are not provided**
- Tested games:
  - Steam: **DOOM**, **Quake**, **Need for Speed**
  - Several **Roblox** games
- **System setup**:
  - Inference server: Linux or WSL
  - Game + Recap: Windows
- **Tested hardware**:
  - Windows 11
  - RTX 5090 (model inference)
  - RTX 5080 (game rendering)

⚠️ **Latency requirement**  
End-to-end inference latency should be **< 50 ms** to avoid performance degradation.

---

### (Optional) WSL Setup

WSL is **only required** if you want to interact with real games.  
You must be on a **Windows machine** to use Recap.

---

### Setup

#### Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Clone the repository
```bash
git clone https://github.com/open-p2p
cd open-p2p
```
All dependencies are managed by uv.

## Geting Started

### Download Model Checkpoints
```bash
uv run python scripts/download_checkpoints.py <150M|300M|600M|1200M>
```

### Download Sample Dataset
```bash
uv run python scripts/download_data.py
```
(We are working on releasing the full dataset)

## Training
We tested training on 8× H100 GPUs. To reproduce the models from paper, use one of the provided configs
- config/policy_model/150M.yaml
- config/policy_model/300M.yaml
- config/policy_model/600M.yaml
- config/policy_model/1200M.yaml
Simply start training by
```bash
uv run elefant/policy_model/train.py --config config/policy_model/150M.yaml
```

### Validation
Training and validation are deliberately separated for stability.

You *can* merge them by lowering `validation_step_interval` in the config, but this may cause instability or crashes.

To run validation on all checkpoints in a directory:
```bash
uv run elefant/policy_model/validation.py --checkpoint_dir <PATH_TO_CHECKPOINT_DIR>
```
Validation results (perplexity) are reported to *Weights & Biases*.

## Inference

Inference can run on Linux **without a display, but real-time game interaction requires Windows + Recap**.

### Start the Inference Server
(On Linux or WSL)
Ensure `model_config.yaml` and `checkpoint.ckpt` are downloaded from Hugging Face.
```bash
uv run elefant/policy_model/inference.py \
  --config checkpoints/150M/model_config.yaml \
  --checkpoint_path checkpoints/150M/checkpoint-step=00500000.ckpt
```
The inference server listens on:
```bash
/tmp/uds.recap
```
This path is automatically detected by **Recap**.

### Start Recap (Windows)
Recap connects the inference server to keyboard and mouse control:
- Captures screenshots from a selected window
- Sends frames to the inference server
- Receives predicted actions
- Executes keyboard and mouse inputs in real time

#### How to Control Recap
1. Select the game window to interact with
2. Ensure the inference server is running at /tmp/uds.recap
3. Press Shift + ]: You should hear a beep: “start capturing with inference”
4. (Move the mouse or press any key to interrupt inference)
5. Press Shift + ] again to properly stop the session
After stopping, a folder will open containing:
- An .mp4 gameplay recording
- An annotation.proto file with recorded keyboard and mouse actions

## Paper & Citation
Coming soon. 