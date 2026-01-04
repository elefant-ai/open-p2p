# Open Pixel2Play (P2P)

**Open Pixel2Play (P2P)** is an open foundation model trained to play video games in real time. The model takes **visual input (images) and text instructions** and outputs **keyboard and mouse actions**, enabling direct interaction with real game environments.

P2P is trained on **8,000+ hours of human-annotated gameplay videos**. We are actively working on releasing the full dataset. In the meantime, a **toy sample dataset** is available on Hugging Face:  
👉 https://huggingface.co/datasets/guaguaa/p2p-toy-examples

Our smallest model (**150M parameters**) can be trained on **8× H100 GPUs in ~70 hours**.

This repository contains:
- The full **training pipeline** for P2P models  
- **Inference code** for running trained models  
- Integration with [**Recap**](https://github.com/elefant-ai/recap) for real-time interaction with commercial games on Windows machine. 

---

## Repository Overview

This repo provides everything needed to:
- Train P2P models from scratch
- Run offline validation
- Serve models for real-time inference
- Connect models to real games via the **Recap** system

---

## Installation
First clone the repo
```bash
git clone https://github.com/open-p2p
cd open-p2p
```

### For Training and offline inference (Linux)

#### Prerequisite \label{requirement}
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
sudo apt update 
sudo apt install build-essential git nvtop htop
sudo add-apt-repository ppa:ubuntuhandbook1/ffmpeg7
sudo apt install ffmpeg
sudo apt install -y libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libavdevice-dev libavfilter-dev
sudo apt install -y clang libclang-dev
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
sudo apt install socat
```

#### Getting Started

##### Download Model Checkpoints
```bash
uv run python scripts/download_checkpoints.py <150M|300M|600M|1200M>
```

##### Download Sample Dataset
```bash
uv run python scripts/download_data.py
```
(We are working on releasing the full dataset)



Training: 

All the dependencies will be handled by `uv`. 

The fully training jobs were completed with 8× H100 GPUs, however the config should work with one H100 or A100 GPU too. 
To reproduce the models from paper, use one of the provided configs (you need to adjust the dataloader related parameters according to your environment for an memory optimized run)
- config/policy_model/150M.yaml
- config/policy_model/300M.yaml
- config/policy_model/600M.yaml
- config/policy_model/1200M.yaml
```bash
uv run elefant/policy_model/train.py --config config/policy_model/150M.yaml
```

Validation:
Training and validation are deliberately separated for stability.

You *can* merge them by lowering `validation_step_interval` in the config, but this may cause instability or crashes.

To run validation on all checkpoints in a directory:
```bash
uv run elefant/policy_model/validation.py --checkpoint_dir <PATH_TO_CHECKPOINT_DIR>
```
Validation results (perplexity) are reported to *Weights & Biases*.

Inference

>**Note**: Inference can run on Linux **without a display, but real-time game interaction requires a Windows machine + Recap**.
1. you probably need to login to your huggingface account as gemme tokenizer needs authentification to dowload
```bash
uv run huggingface-cli login
```
To validate the inference, if you already download the checkpoint and model config you can run 
```bash
uv run elefant/policy_model/inference.py \
  --config checkpoints/150M/model_config.yaml \
  --checkpoint_path checkpoints/150M/checkpoint-step=00500000.ckpt
```
or if you just want to check the code works, run
```bash
uv run elefant/policy_model/inference.py --config=config/policy_model/150M.yaml --use_random_weights
```



### For running inference to interact with game in real-time (Windows)

Note that **Game environments are not provided**
- Games used to test model real-time playing:
  - Steam: **DOOM**, **Quake**, **Need for Speed**
    - Quake: **Mouse sensitivity**: 3.5, **smoothing**: disabled
    - DOOM: **smoothing**: 2x, **look sensitivity**: 22%, **move sensitivity**: 22%
  - Several **Roblox**: **Rivals**, **Natural Survival Disaster**, **Hypershot**, **Be a Shark**, **Blade Ball**, and etc. 
    - **Camera sensitivity**: 0.52.
- **Tested hardware**:
  - Windows 11
  - RTX 5090 (model inference)
  - RTX 5080 (game rendering)

#### Latency requirement
Any hardware that achieves an end-to-end inference latency of < 50 ms should be sufficient.
A detailed latency breakdown is provided by **Recap** [Latency Analysis](assets/latency.png), this chart will be generated once you finish a model inference session. 

---

#### Prerequisite 
0.0 Install Nvidia GPU driver

0. Please first follow the instruction to install [**Recap**](https://github.com/elefant-ai/recap)

1. Install WSL on Window machine:

#### 1. Install WSL with Ubuntu 24.04
```bash
wsl --install -d Ubuntu-24.04
```
Reboot if prompted.

#### 2. Increase WSL Memory Limit (Recommended)
Create or edit the file:
```bash
C:\Users\<username>\.wslconfig
```
Add
```bash
[wsl2]
memory=52GB
```
Set this to a large fraction of your system RAM.
Restart WSL (or reboot) for changes to take effect.

#### 3. Install Core Dependencies (Inside WSL)
Follow the instruction on \ref{requirement}


### Getting started

1. Start the Inference Server (On WSL environment)
Ensure `model_config.yaml` and `checkpoint.ckpt` are downloaded from Hugging Face.

Without Text Input (Default)

This runs the model **without textual instructions**.

Due to compilation constraints, text input cannot be enabled or disabled at runtime and the mode must be chosen at launch. A model started without text input cannot accept text later.

The majority of the experiments in our paper use the no-text-input setting.
```bash
uv run elefant/policy_model/inference.py \
  --config checkpoints/150M/model_config.yaml \
  --checkpoint_path checkpoints/150M/checkpoint-step=00500000.ckpt
```

With Text Input

This runs the model with text instructions enabled.
```bash
uv run elefant/policy_model/inference.py \
  --config checkpoints/150M/model_config.yaml \
  --checkpoint_path checkpoints/150M/checkpoint-step=00500000.ckpt
  --input_text
```

The inference server listens on:
```bash
/tmp/uds.recap
```
This path is automatically detected by **Recap**.

2. Start Recap (Use Windows Command Prompt)
CD to recap repo and run `just trace` on the command prompt, you should see [UI](assets/UI.png). Fill in `you name` and `env` boxes which will be filled in as metadata in the annotation proto that's to be generated. 

2.1 Choose the right window to capture (such as Roblox on [UI](assets/UI.png))
2.3. Press `Shift` + `]`: You should hear a beep: “start capturing with inference”
2.4. (Move the mouse or press any key to interrupt inference, then press `[` to resume model controlling)
2.5. Press `Shift` + `]`again to properly stop the session
After stopping, a [folder](assets/folder.png) will open containing:
- An .mp4 gameplay recording
- An annotation.proto file with recorded keyboard and mouse actions

> Note: Recap connects the inference server to keyboard and mouse control:
- Captures screenshots from a selected window
- Sends frames to the inference server
- Receives predicted actions
- Executes keyboard and mouse inputs in real time


## Paper & Citation
Coming soon. 