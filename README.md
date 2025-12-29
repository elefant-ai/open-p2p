## Training

To run training, download data files locally and then simply run:

`uv run elefant/policy_model/train.py --config config/policy_model/dev.yaml`

## Validation

We don't do validation during training, it is done separately from it. To run validation you can simply run the below command with the checkpoint directory indicating saved checkpoints. It will run validation and report perplexity to wandb for all the checkpoints present in the folder

`uv run elefant/policy_model/validation.py --checkpoint_dir checkpoint/directory`

## Inference

For running inference with random weights:

`uv run elefant/policy_model/inference.py --config config/policy_model/dev.yaml --use_random_weights`

For running inference with non random weights:

`uv run elefant/policy_model/inference.py --config config/policy_model/dev.yaml --checkpoint_path model_checkpoint.ckpt`