import argparse
from elefant.policy_model.config import LightningPolicyConfig
from elefant.config import load_config
from elefant.policy_model.stage3_finetune import train_stage3_finetune
import logging
import torch
import elefant.torch


def lightning_main():
    logging.basicConfig(level=logging.INFO, force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--data_folder", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config, LightningPolicyConfig)
    config.shared.fast_dev_run = args.fast_dev_run

    if args.data_folder is not None:
        logging.info(f"Using local data folder: {args.data_folder}")
        config.stage3_finetune.training_dataset.local_prefix = args.data_folder
        for val_dataset in config.stage3_finetune.validation_datasets:
            val_dataset.local_prefix = args.data_folder

    if args.fast_dev_run:
        logging.warning("!!!Fast dev run is enabled!!!")

    if args.no_compile:
        logging.warning("!!!No compile is enabled!!!")
        torch.compiler.set_stance("force_eager")

    elefant.torch.pytorch_setup()
    train_stage3_finetune(config)


if __name__ == "__main__":
    lightning_main()
