from typing import Optional, List, Literal

import pydantic
from elefant.config import ConfigBase
from elefant.data import (
    RandAugmentationConfig,
    UniversalAutoregressiveActionMappingConfig,
)
from elefant.config import WandbConfig
from elefant.im_tokenizer.config import ImageTokenizerConfig
from elefant.text_tokenizer.config import TextTokenizerConfig


class DatasetConfig(ConfigBase):
    local_prefix: str = ""
    shuffle: bool = True
    n_preprocess_threads_per_gpu: int = 16
    preprocessed_chunks_queue_size_per_gpu: int = 4096

    warn_on_starvation: bool = False

    rand_augmentation: RandAugmentationConfig = RandAugmentationConfig()
    shuffle_buffer_size_per_gpu: int = 1024
    always_labelled: bool = False

    batch_size: int = 128
    dataset_worker_prefetch_factor: int = 1
    dataset_worker_num_workers_per_gpu: int = 1
    shuffled_chunks_queue_size_per_gpu: int = 16
    ## validation type used only for validation dataloader
    ## for example when we only want to validate
    ## action grounding head we need to use only the labelled data

    n_seq_timesteps: int = 3
    tokenizer: ImageTokenizerConfig = pydantic.Field(default=ImageTokenizerConfig())
    # See https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.trainer.trainer.Trainer.html
    # generally either "bf16-mixed" or "32-true"

    # text related
    text_annotation_model_version: List[str] = pydantic.Field(
        default=["gemini-2.5-flash", "gemini-2.5-flash-thinking-0905"]
    )


class ValidationDatasetConfig(DatasetConfig):
    validation_name: str


class ActionLabelDatasetConfig(DatasetConfig):
    pass


class SparseMoEConfig(ConfigBase):
    num_experts: int = 16
    experts_per_token: int = 4
    lb_loss_weight: float = 0.01
    rz_loss_weight: float = 0.001


class ActionDecoderConfig(ConfigBase):
    embed_dim: int = 256
    n_q_head: int = 8
    n_kv_head: int = 8
    n_transformer_layers: int = 3


class TransformerModelConfig(ConfigBase):
    transformer_dim: int = 128
    n_transformer_layers: int = 4
    n_q_head: int = 8
    n_kv_head: int = 8
    n_thinking_tokens: int = 1
    model_type: Literal["dense", "sparse_moe"] = "dense"
    sparse_moe: SparseMoEConfig = SparseMoEConfig()
    action_decoder: ActionDecoderConfig = ActionDecoderConfig()


class PolicyModelConfig(TransformerModelConfig):
    mask_block_size: Optional[int] = None
    attention_history_len: List[int] = [
        100,
        100,
        100,
        100,
    ]  # should match n_transformer_layers
    n_kv_sink_tokens: int = 1
    top_p: Optional[float] = None
    z_loss_weight: float = 1e-4


class AdamWOptimConfig(ConfigBase):
    learning_rate: float = 3e-4

    weight_decay: float = 1e-2
    beta_1: float = 0.95
    beta_2: float = 0.999


class Stage3InitConfig(ConfigBase):
    """3 options for stage3.

    random, initalize with stage2 model from this config, initialize with stage2 model from different config.

    if random, then we just initialize with random weights.
    if stage2_model_path, then we initialize with stage2 model from this path, otherwise find the stage2 model from this config.
    """

    random: bool = False
    stage2_model_path: Optional[str] = pydantic.Field(
        default=None,
        description="If loading checkpoints from a different path from the output path this is set. It should point directly at a checkpoint file, not a folder.",
    )
    stage3_model_path: Optional[str] = pydantic.Field(
        default=None,
        description="If loading checkpoints from a different path from the output path this is set. It should point directly at a checkpoint file, not a folder.",
    )


class PolicyTrainingConfig(ConfigBase):
    n_training_steps: int = pydantic.Field(
        default=100_000,
        description="How many training steps to run.",
    )
    optim: AdamWOptimConfig = pydantic.Field(default=AdamWOptimConfig())

    accumulate_grad_batches: int = pydantic.Field(
        default=1,
        description="How many training steps between optimizer steps.",
    )

    training_dataset: ActionLabelDatasetConfig = pydantic.Field(
        default=ActionLabelDatasetConfig(shuffle=True),
    )

    validation_datasets: List[ValidationDatasetConfig] = pydantic.Field(
        default=[],
    )

    validation_step_interval: int = pydantic.Field(
        default=100,
        description="How many training steps between validation steps.",
    )
    n_validation_steps: int = pydantic.Field(
        default=10,
        description="How many validation steps to run.",
    )

    save_every_n_steps: int = pydantic.Field(
        default=1000,
        description="How many training steps between checkpoints.",
    )

    freeze_transformer_layers_for_steps: int = pydantic.Field(
        default=0,
        description="If non-zero, the transformer layers are frozen for the first N steps.",
    )


class Stage3FineTuneConfig(PolicyTrainingConfig):
    init: Stage3InitConfig = pydantic.Field(default=Stage3InitConfig())


class SharedConfig(ConfigBase):
    # Note, this will be overridden by the command line (it should never be set in a config anyway).
    fast_dev_run: bool = False
    frame_height: int = 192
    frame_width: int = 192
    output_path: str = pydantic.Field(default="tmp")

    # This is the maximum number of steps (image, action) pairs in a single sequence.
    n_seq_timesteps: int = 3
    tokenizer: ImageTokenizerConfig = pydantic.Field(default=ImageTokenizerConfig())
    # See https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.trainer.trainer.Trainer.html
    # generally either "bf16-mixed" or "32-true"
    precision: str = "bf16-mixed"
    text_tokenizer_config: TextTokenizerConfig = TextTokenizerConfig()
    action_mapping: UniversalAutoregressiveActionMappingConfig = (
        UniversalAutoregressiveActionMappingConfig()
    )


class InferenceConfig(ConfigBase):
    checkpoint_path: Optional[str] = pydantic.Field(
        default=None,
        description="If loading checkpoints from a different path from the output path this is set. It should point directly at a checkpoint file, not a folder.",
    )
    mouse_sampling_approach: str = pydantic.Field(
        default="mean",
        description="How to sample the action from the model's output.",
    )
    sampling_temperature: float = pydantic.Field(
        default=1.0,
        description="The temperature to use for sampling the action from the model's output.",
    )


class LightningPolicyConfig(ConfigBase):
    shared: SharedConfig = pydantic.Field(default=SharedConfig())
    policy_model: PolicyModelConfig = pydantic.Field(default=PolicyModelConfig())
    stage3_finetune: Stage3FineTuneConfig = pydantic.Field(
        default=Stage3FineTuneConfig()
    )
    wandb: WandbConfig = pydantic.Field(default=WandbConfig())
    inference: InferenceConfig = pydantic.Field(default=InferenceConfig())
