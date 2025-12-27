## Training

To run training, download data files locally and then simply run:

`uv run elefant/policy_model/train.py --config config/policy_model/dev.yaml`

## Inference

For running inference with random weights:

`uv run elefant/policy_model/inference.py --config config/policy_model/dev.yaml --use_random_weights`

For running inference with non random weights:

`uv run elefant/policy_model/inference.py --config config/policy_model/dev.yaml --checkpoint_path model_checkpoint.ckpt`