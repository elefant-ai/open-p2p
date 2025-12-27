import logging
from typing import NamedTuple, Optional, List

import torch
from torch.utils.data.dataloader import default_collate

from elefant.data.action_mapping import (
    UniversalAutoregressiveActionMapping,
    UniversalAutoregressiveActionMappingConfig,
)
from elefant.data.environment_mapping import ENVIRONMENT_MAPPING
from elefant.data.video_proto_dataset import (
    ProtoParser,
    VideoProtoDataset,
    VideoProtoDatasetConfig,
)
from elefant.data.proto import shared_pb2
from elefant.text_tokenizer.config import TextTokenizerConfig
from elefant.data.proto import video_annotation_pb2


class ActionLabelVideoProtoDatasetConfig(VideoProtoDatasetConfig):
    action_mapping: UniversalAutoregressiveActionMappingConfig = (
        UniversalAutoregressiveActionMappingConfig()
    )
    drop_chunks_with_only_system_actions: bool = False
    text_tokenizer_config: Optional[TextTokenizerConfig] = None


class ActionLabelVideoDatasetItem(NamedTuple):
    frames: torch.Tensor  # (T, H, W, C)
    action_annotations: torch.Tensor  # (T, action_seq_len)
    env_subenv_encoding: torch.Tensor  # (T)
    # A binary mask indicating whether the action was by the user (if mask is 0 then it is a system action or unknown)
    user_action_mask: torch.Tensor  # (T)
    text_embeddings: torch.Tensor  # (T, n_text_tokens, text_embedding_dim)
    system_action_mask: torch.Tensor  # (T)
    valid_frame_mask: torch.Tensor | None = None


class ActionLabelAnnotationParser(ProtoParser):
    def __init__(
        self,
        proto_local_path: str,
        always_labelled: bool,
        n_frames: int,
        action_mapping: UniversalAutoregressiveActionMapping,
        drop_chunks_with_only_system_actions: bool,
        text_annotation_model_version: List[str] | None,
        text_tokenizer_name: str | None,
        text_embedding_shape: List[int] | None,
    ):
        super().__init__(proto_local_path, always_labelled, n_frames)
        self.action_mapping = action_mapping
        self.action_seq_len = action_mapping.get_seq_len()
        self.drop_chunks_with_only_system_actions = drop_chunks_with_only_system_actions
        self.text_annotation_model_version = text_annotation_model_version
        self.text_tokenizer_name = text_tokenizer_name
        self.text_embedding_shape = text_embedding_shape
        if self.frame_annotations is not None:
            if len(self.frame_annotations) != n_frames and n_frames != -1:
                if always_labelled:
                    raise ValueError(
                        f"Number of frames in the proto ({len(self.frame_annotations)}) does not match the number of frames in the dataset ({n_frames})"
                    )
                else:
                    # we have some legacy unlabelled data that has more annotations than frames, that's fine because all the annotations will be rewritten during training
                    logging.debug(
                        f"Number of frames in the proto ({len(self.frame_annotations)}) does not match the number of frames in the dataset ({n_frames})"
                    )

    def get_repeated_text_embedding(
        self,
        frame_text_annotation: video_annotation_pb2.FrameTextAnnotation,
        text_embeddings: torch.Tensor,
        idx: int,
        n_frames: int,
    ) -> torch.Tensor:
        """
        this function will repeat text embedding for all the frames with the duration of the text annotation
        in the frame_text_annotation, we only have text annotation at the first frame, and we want to
        impute all the frames for the duration of the text annotation.

        frame_text_annotation is a list because each frame can have text annotations from multiple models.
        """
        if self.text_tokenizer_name is None:
            logging.info("No text tokenizer name provided, skipping text embedding")
            return text_embeddings

        for text_annotation in frame_text_annotation:
            if (
                text_annotation.frame_text_annotator.version
                in self.text_annotation_model_version
            ):
                this_text_embedding = torch.tensor(
                    list(
                        text_annotation.text_embedding_dict[
                            text_annotation.frame_text_annotator.version
                        ]
                        .text_embeddings[self.text_tokenizer_name]
                        .values
                    )
                )
                shape = list(
                    text_annotation.text_embedding_dict[
                        text_annotation.frame_text_annotator.version
                    ]
                    .text_embeddings[self.text_tokenizer_name]
                    .shape
                )
                if shape == []:
                    logging.warning(
                        "This happened when text annotation is presented but precomputed text embedding is not available"
                    )
                    # TODO: this is because the text embedding is not precomputed for
                    # this example, this should not happen if all the videos
                    # are precomputed, so this should be removed later.
                    return text_embeddings
                this_text_embedding = this_text_embedding.reshape(shape)
                assert text_annotation.duration > 0, "Duration must be greater than 0"
                last_frames = int(text_annotation.duration * 20)
                if idx + last_frames > n_frames:
                    num_embedding_to_impute = n_frames - idx
                else:
                    num_embedding_to_impute = last_frames
                text_embeddings[idx : (idx + num_embedding_to_impute), :, :] = (
                    this_text_embedding.expand(num_embedding_to_impute, -1, -1)
                )
                break
        return text_embeddings

    def annotate_frames(
        self,
        frames: torch.Tensor,
        start_frame: int,
        end_frame: int,
        valid_frame_mask: torch.Tensor,
    ) -> ActionLabelVideoDatasetItem:
        action_annotations = []
        n_frames = frames.shape[0]
        ## fill in unknown envs with -1
        try:
            env_subenv_encoding = ENVIRONMENT_MAPPING[self.metadata.env.env][
                self.metadata.env.env_subtype
            ]
        except (AttributeError, KeyError):
            env_subenv_encoding = -1

        env_subenv_encoding = torch.full(
            size=(n_frames,), fill_value=env_subenv_encoding, dtype=torch.long
        )
        if self.frame_annotations is not None:
            action_annotations = self.action_mapping.make_empty_action(n_frames)
            user_action_mask = torch.zeros(n_frames, dtype=torch.bool)
            system_action_mask = torch.zeros(n_frames, dtype=torch.bool)
            text_embeddings = torch.zeros(n_frames, *self.text_embedding_shape)
            for i, frame_annotation in enumerate(
                self.frame_annotations[start_frame:end_frame]
            ):
                if self.text_tokenizer_name is not None:
                    text_embeddings = self.get_repeated_text_embedding(
                        frame_annotation.frame_text_annotation,
                        text_embeddings,
                        i,
                        n_frames,
                    )

                user_action = frame_annotation.user_action
                system_action = frame_annotation.system_action
                if system_action.is_known:
                    user_action_mask[i] = False
                    system_action_mask[i] = True
                    keys = list(system_action.keyboard.keys)
                    mouse_buttons = list(system_action.mouse.buttons_down)
                    mouse_delta_px = system_action.mouse.mouse_delta_px
                    if user_action.is_known:
                        logging.warning(
                            f"User and system actions are both known for frame {i}"
                        )
                elif user_action.is_known:
                    user_action_mask[i] = True
                    system_action_mask[i] = False
                    keys = list(user_action.keyboard.keys)
                    mouse_buttons = list(user_action.mouse.buttons_down)
                    mouse_delta_px = user_action.mouse.mouse_delta_px
                else:
                    # Make a default action, no known user action.
                    user_action_mask[i] = False
                    system_action_mask[i] = False
                    keys = []
                    mouse_buttons = []
                    mouse_delta_px = shared_pb2.Vec2Int()

                this_action = self.action_mapping.action_to_tensor(
                    keys=keys,
                    mouse_buttons=mouse_buttons,
                    mouse_delta_px=mouse_delta_px,
                )
                for n in ["keys", "mouse_buttons", "mouse_delta_x", "mouse_delta_y"]:
                    getattr(action_annotations, n)[i, :] = getattr(this_action, n)

            if self.drop_chunks_with_only_system_actions and system_action_mask.all():
                return None

            # If the user or system action is known, then it should be a valid frame.
            assert torch.all(
                valid_frame_mask >= (user_action_mask | system_action_mask)
            )

            return ActionLabelVideoDatasetItem(
                frames=frames,
                action_annotations=action_annotations,
                env_subenv_encoding=env_subenv_encoding,
                user_action_mask=user_action_mask,
                system_action_mask=system_action_mask,
                valid_frame_mask=valid_frame_mask,
                text_embeddings=text_embeddings,
            )
        else:
            logging.warning("Frame annotations are None - this should not happen")
            return None


class ActionLabelVideoProtoDataset(VideoProtoDataset):
    """This is for dataset (e.g. for learning latent actions)
    where either the action label is not available or not used.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataloader = None
        self.action_mapping = UniversalAutoregressiveActionMapping(
            config=self.config.action_mapping
        )

    def _proto_parser_factory(
        self, proto_local_path: str, always_labelled: bool, n_frames: int
    ) -> ProtoParser:
        if self.config.text_tokenizer_config is None:
            text_annotation_model_version = None
            text_tokenizer_name = None
            text_embedding_shape = None
        else:
            text_annotation_model_version = (
                self.config.text_tokenizer_config.text_annotation_model_version
            )
            text_tokenizer_name = self.config.text_tokenizer_config.text_tokenizer_name
            text_embedding_shape = (
                self.config.text_tokenizer_config.text_embedding_shape
            )
        return ActionLabelAnnotationParser(
            proto_local_path,
            always_labelled,
            n_frames,
            self.action_mapping,
            self.config.drop_chunks_with_only_system_actions,
            text_annotation_model_version,
            text_tokenizer_name,
            text_embedding_shape,
        )


if __name__ == "__main__":
    ds_cfg = ActionLabelVideoProtoDatasetConfig(
        local_prefix="/path/to/your/data/folder",
        frame_height=192,
        frame_width=192,
        T=200,
        shuffle=False,
        batch_size=1,
        n_preprocess_workers_per_iter_worker=1,
        warn_on_starvation=False,
        dataset_worker_prefetch_factor=1,
        dataset_worker_num_workers=1,
        dataset_unique_id="unique_dataset",
        text_tokenizer_config=TextTokenizerConfig(
            text_tokenizer_name="gemma",
            text_embedding_shape=[1, 768],
            text_annotation_model_version=[
                "gemini-2.5-flash",
                "gemini-2.5-flash-thinking-0905",
            ],
        ),
    )
    dataset = ActionLabelVideoProtoDataset(ds_cfg, device="cpu")
    dataloader = dataset.to_dataloader()
    for item in dataloader:
        import ipdb

        ipdb.set_trace()
