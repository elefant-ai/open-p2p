## Download Toy Example

Download the toy training dataset, which contains 3 examples:

```bash
uv run huggingface-cli download guaguaa/p2p-toy-examples toy-examples.zip \
  --repo-type dataset \
  --local-dir .
unzip toy-examples.zip -d .
rm toy-examples.zip
```

## Training

To start training, run the following command.  
Note: the compilation step may take some time. The provided `dev.yaml` configuration has been tested on an A100 GPU.

```bash
uv run elefant/policy_model/train.py \
  --config config/policy_model/dev.yaml
```

## Inference
1. First, download the model checkpoints, the checkpoints for 150M and 1.2B models are available now
```bash
uv run huggingface-cli download guaguaa/p2p-150M --local-dir ./checkpoints/150M/
```

```bash
uv run huggingface-cli download guaguaa/p2p-1.2B --local-dir ./checkpoints/1200M/
```

Run inference with a trained checkpoints:

```bash
uv run elefant/policy_model/inference.py \
  --config=checkpoints/150M/model_config.yaml  \
  --checkpoint_path checkpoints/150M/checkpoint-step=00500000.ckpt
```

The inference server will run at `/tmp/uds.recap`. This path is automatically picked up by **Recap** if you want to interact with the model in a real game environment.  

> Note: Inference has only been tested on an RTX 5090 GPU using a Windows machine with WSL.

### How to Use with Recap

1. Run Recap in a Windows terminal.  
2. Start the inference server in WSL and wait until it is ready.  
3. Start Recap model inference:
   - Press `Shift + ]` to start.
   - Press any key to interrupt.
   - Press `Shift + ]` again to resume.
