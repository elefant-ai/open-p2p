"""
Example on how to run (Only need to indicate the checkpoint directory on r2):

If you want to run the validation on already trained model:

uv run elefant/policy_model/validation.py --checkpoint_dir iraki/policy/dev/6/stage3_finetune


If you want to run the validation alongside training:

uv run elefant/policy_model/validation.py \
  --checkpoint_dir iraki/policy/dev/6/stage3_finetune \
  --watch_for_checkpoints

To restrict validation to a range of steps (inclusive):

uv run elefant/policy_model/validation.py \
  --checkpoint_dir iraki/policy/dev/6/stage3_finetune \
  --min_steps 40000 \
  --max_steps 100000
"""

import re
import argparse
import os
import fsspec
import wandb
import logging
import shutil
import torch
import time

from typing import Optional
from datetime import datetime

from elefant.policy_model.utils import load_config_from_checkpoint
from elefant.policy_model.stage3_finetune import (
    Stage3DataModule,
    Stage3LabelledBCLightning,
)
from elefant.data import ActionLabelVideoDatasetItem, StructuredAction
from elefant.metrics import LossMetric
from elefant.torch import cross_entropy_to_perplexity


def move_action_label_video_dataset_item_to_device(
    batch, device: torch.device, dtype: torch.dtype = None
):
    """Moves the tensors within an ActionLabelVideoDatasetItem to the specified device and dtype."""
    moved_frames = batch.frames.to(device)

    moved_action_annotations = StructuredAction(
        keys=batch.action_annotations.keys.to(device),
        mouse_buttons=batch.action_annotations.mouse_buttons.to(device),
        mouse_delta_x=batch.action_annotations.mouse_delta_x.to(device),
        mouse_delta_y=batch.action_annotations.mouse_delta_y.to(device),
    )
    moved_env_subenv_encoding = batch.env_subenv_encoding.to(device)
    moved_user_action_mask = batch.user_action_mask.to(device)
    moved_system_action_mask = batch.system_action_mask.to(device)
    moved_valid_frame_mask = batch.valid_frame_mask.to(device)
    moved_text_embeddings = batch.text_embeddings.to(device)
    return ActionLabelVideoDatasetItem(
        frames=moved_frames,
        action_annotations=moved_action_annotations,
        env_subenv_encoding=moved_env_subenv_encoding,
        user_action_mask=moved_user_action_mask,
        system_action_mask=moved_system_action_mask,
        valid_frame_mask=moved_valid_frame_mask,
        text_embeddings=moved_text_embeddings,
    )


def find_all_checkpoints(path: str):
    if path.endswith(".ckpt"):
        return [(path, datetime(1900, 1, 1).replace(tzinfo=None))]

    fs, root = fsspec.url_to_fs(path)
    fs.invalidate_cache()
    try:
        entries = fs.ls(root, detail=True)
    except FileNotFoundError:
        return []

    ckpt_entries = []
    for e in entries:
        name = e.get("name") or e.get("Key") or e.get("path")
        if not name or not name.endswith(".ckpt"):
            continue

        proto = fs.protocol
        if isinstance(proto, (list, tuple)):
            proto = proto[0] if proto else None

        if proto and proto != "file" and "://" not in name:
            ckpt_path = f"{proto}://{name}"
        else:
            ckpt_path = name

        try:
            step = extract_step_from_checkpoint_path(ckpt_path)
        except ValueError:
            continue

        last_modified = e.get("LastModified")
        if hasattr(last_modified, "replace"):
            last_modified = last_modified.replace(tzinfo=None)
        else:
            last_modified = datetime(1900, 1, 1).replace(tzinfo=None)

        ckpt_entries.append((step, ckpt_path, last_modified))

    # sort by step ascending
    ckpt_entries.sort(key=lambda t: t[0])
    return [(p, mtime) for _, p, mtime in ckpt_entries]


def extract_step_from_checkpoint_path(checkpoint_path):
    match = re.search(r"step=(\d+)", checkpoint_path)

    if match:
        # group(1) contains the captured digits. int() handles leading zeros.
        global_step = int(match.group(1))
        print(f"Successfully extracted global_step: {global_step}")
    else:
        raise ValueError(
            f"Could not find global_step in the checkpoint path: {checkpoint_path}"
        )
    return global_step


def set_validation_dataset_cfg_to_single_thread(validation_dataset_cfgs, batch_size):
    # low numbers to not run out of space since the machine we run validation on is small and has small storage
    validation_dataset_cfgs.n_preprocess_threads_per_gpu = 2
    validation_dataset_cfgs.preprocessed_chunks_queue_size_per_gpu = 1
    validation_dataset_cfgs.dataset_worker_prefetch_factor = 2
    validation_dataset_cfgs.batch_size = batch_size
    return validation_dataset_cfgs


def report_validation_metrics(checkpoint_path, global_step, run_id: str):
    BATCH_SIZE_FOR_VAL = 1
    t0 = time.time()
    checkpoint_path = checkpoint_path.replace("research-training-checkpoints/", "")

    try:
        config = load_config_from_checkpoint(checkpoint_path)
        stage = os.path.basename(os.path.dirname(checkpoint_path))
        print(f"validating stage {stage}, run id {run_id}, global step {global_step}")
        wandb_kwargs = dict(
            entity="elefantai",
            project=config.wandb.project,
            group=config.wandb.exp_name,
            name=config.wandb.exp_name + "_validation",
            job_type="validation",
        )
        wandb_kwargs.update(dict(id=run_id, resume="allow"))
        wandb.init(**wandb_kwargs)
        wandb.define_metric("trainer/global_step")
        wandb.define_metric("*", step_metric="trainer/global_step")
        # prepare dataset
        datamodule = Stage3DataModule(config)
        for idx in range(len(config.stage3_finetune.validation_datasets)):
            config.stage3_finetune.validation_datasets[idx] = (
                set_validation_dataset_cfg_to_single_thread(
                    config.stage3_finetune.validation_datasets[idx],
                    BATCH_SIZE_FOR_VAL,
                )
            )
        model = Stage3LabelledBCLightning.load_from_checkpoint(
            checkpoint_path, config=config, inference_mode=True
        )
        batch_size = config.stage3_finetune.validation_datasets[-1].batch_size
        n_validation_steps = config.stage3_finetune.n_validation_steps

        n_validation_steps = n_validation_steps * batch_size // BATCH_SIZE_FOR_VAL * 8

        datamodule.setup(stage)
        val_dataloaders = datamodule.val_dataloader()
        del datamodule

        model.eval()
        model = model.to("cuda")

        # Ensure all block_masks are moved to the correct device
        # This is needed because block_masks might be created on CPU during model initialization
        if hasattr(model, "bc_transformer") and hasattr(
            model.bc_transformer, "block_mask_to_device"
        ):
            model.bc_transformer.block_mask_to_device(model.device)

        print(f"takes {time.time() - t0:.3f}s to prepare the dataset and model")
        # Get the model's dtype for input tensor conversion
        validation_metrics = {}
        model_dtype = next(model.parameters()).dtype
        action_types = list(StructuredAction._fields)
        action_type_to_metric_name = {}
        for field_name in action_types:
            if field_name == "keys":
                action_type_to_metric_name[field_name] = "key"
            elif field_name == "mouse_buttons":
                action_type_to_metric_name[field_name] = "mouse_button"
            else:
                action_type_to_metric_name[field_name] = field_name

        for val_set_name in val_dataloaders.keys():
            metrics = {
                "off_perplexity": LossMetric().to(model.device),
            }
            for action_type in action_types:
                metric_name = action_type_to_metric_name[action_type]
                metrics[f"off_perplexity_{metric_name}"] = LossMetric().to(model.device)

            validation_metrics[val_set_name] = metrics

        start_time = time.time()
        for val_set_name, val_dataloader in val_dataloaders.items():
            print(f"\nProcessing validation set: {val_set_name}")
            val_metrics = validation_metrics[val_set_name]
            for batch_idx, batch in enumerate(val_dataloader):
                batch_to_cuda = move_action_label_video_dataset_item_to_device(
                    batch, "cuda", dtype=model_dtype
                )

                actions_in, masked_labels, _ = model._create_target_and_masked_labels(
                    batch_to_cuda
                )

                with (
                    torch.inference_mode(),
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16),
                ):
                    loss, _, losses, auxiliary_outputs = model._calculate_loss(
                        batch_to_cuda,
                        actions_in,
                        masked_labels,
                        batch_to_cuda.text_embeddings,
                    )

                val_metrics["off_perplexity"].update(cross_entropy_to_perplexity(loss))
                for k, v in losses.items():
                    if k == "rz_loss" or k == "lb_loss":
                        continue
                    val_metrics[f"off_perplexity_{k}"].update(
                        cross_entropy_to_perplexity(v).item()
                    )

                if batch_idx % 100 == 0:
                    total_time = time.time() - start_time
                    start_time = time.time()
                    print(f"Batch {batch_idx}: Total processing time {total_time:.3f}s")

                if batch_idx >= n_validation_steps:
                    break

        for val_set_name, val_metrics in validation_metrics.items():
            for metric_name, metric in val_metrics.items():
                value = metric.compute()
                if isinstance(value, torch.Tensor):
                    value = value.detach().item()
                print({f"{val_set_name}_validation_{metric_name}": value})

        total_time = time.time() - t0
        print(f"Total validation time: {total_time:.3f}s")

        for val_set_name, val_metrics in validation_metrics.items():
            log_data = {}
            for metric_name, metric in val_metrics.items():
                value = metric.compute()
                if isinstance(value, torch.Tensor):
                    value = value.detach().item()
                log_data[f"{val_set_name}_validation_{metric_name}"] = value
            log_data["trainer/global_step"] = int(global_step)
            wandb.log(log_data, step=int(global_step))
    finally:
        wandb.finish()
        shutil.rmtree("/tmp/elefant_zmq", ignore_errors=True)
        shutil.rmtree("/ephemeral/elefant_tmp_data", ignore_errors=True)
        shutil.rmtree("/tmp/elefant_data", ignore_errors=True)
        logging.info("Cleaned tmp dataset dirs")



def is_step_in_range(
    step: int, min_steps: Optional[int], max_steps: Optional[int]
) -> bool:
    if min_steps is not None and step < min_steps:
        return False
    if max_steps is not None and step > max_steps:
        return False
    return True


def run_validation_or_watcher(
    checkpoint_dir: str,
    watch_for_checkpoints: bool,
    min_steps: Optional[int],
    max_steps: Optional[int],
):
    """
    Local execution path: run once or watch and run repeatedly
    """
    run_id = wandb.util.generate_id()
    if watch_for_checkpoints:
        # track which steps we've already validated during this run
        processed_steps = set()

        while True:
            ckpts = find_all_checkpoints(checkpoint_dir)
            if not ckpts:
                logging.info("No checkpoints found, sleeping for 30 minutes")
                time.sleep(1800)
                continue

            # filter for new steps in range (ascending order already guaranteed by find_all_checkpoints)
            new_ckpts = []
            for checkpoint_path, _ in ckpts:
                try:
                    step = extract_step_from_checkpoint_path(checkpoint_path)
                except ValueError:
                    continue
                if step in processed_steps:
                    continue
                if not is_step_in_range(step, min_steps, max_steps):
                    continue
                new_ckpts.append((checkpoint_path, step))

            if not new_ckpts:
                if min_steps is not None or max_steps is not None:
                    logging.info(
                        "No new checkpoints found within the specified step range, sleeping for 30 minutes"
                    )
                else:
                    logging.info("No new checkpoints found, sleeping for 10 minutes")
                time.sleep(600)
                continue

            for checkpoint_path, global_step in new_ckpts:
                checkpoint_parent = os.path.dirname(checkpoint_path)
                report_validation_metrics(checkpoint_path, global_step, run_id)
                logging.info(
                    f"Validation metrics logged successfully for checkpoint {checkpoint_path}"
                )
                processed_steps.add(global_step)

    else:
        ckpts = find_all_checkpoints(checkpoint_dir)
        if not ckpts:
            logging.info(f"No checkpoints found in {checkpoint_dir}")
        for checkpoint_path, _ in ckpts:
            try:
                global_step = extract_step_from_checkpoint_path(checkpoint_path)
            except ValueError:
                logging.warning(
                    f"Skipping checkpoint with unparseable step: {checkpoint_path}"
                )
                continue

            if not is_step_in_range(global_step, min_steps, max_steps):
                continue

            report_validation_metrics(checkpoint_path, global_step, run_id)
    logging.info("Validation metrics logged successfully.")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Checkpoint dir (or single .ckpt) to validate",
    )
    parser.add_argument(
        "--watch_for_checkpoints",
        action="store_true",
        help="Run in background and continuously monitor for new checkpoints",
    )
    parser.add_argument(
        "--min_steps",
        type=int,
        default=None,
        help="Minimum global step (inclusive) of checkpoints to validate",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum global step (inclusive) of checkpoints to validate",
    )
    args = parser.parse_args()

    # sanity check the range if both are provided
    if args.min_steps is not None and args.max_steps is not None:
        if args.min_steps > args.max_steps:
            parser.error("--min_steps must be <= --max_steps")

    run_validation_or_watcher(
        checkpoint_dir=args.checkpoint_dir,
        watch_for_checkpoints=args.watch_for_checkpoints,
        min_steps=args.min_steps,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
