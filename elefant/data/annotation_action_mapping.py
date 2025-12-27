# Derived from VPT action mapping.
# https://github.com/openai/Video-Pre-Training/blob/main/lib/action_mapping.py
# https://github.com/openai/Video-Pre-Training/blob/main/lib/actions.py

from dataclasses import dataclass
import torch
import numpy as np
from elefant.data.proto import video_annotation_pb2
from elefant.data.policy_action_mapping import CameraHierarchicalMapping
from elefant.data.const import *
import attr


@dataclass
class FactoredActionType:
    buttons: np.ndarray  # [20]
    camera: np.ndarray  # [2]


@dataclass
class JointActionType:
    buttons: np.ndarray  # [121]
    camera: np.ndarray  # [8641]


class SyntheticButtons:
    # Composite / scripted actions
    CHANNEL_ATTACK = "channel-attack"

    ALL = [CHANNEL_ATTACK]


class QuantizationScheme:
    LINEAR = "linear"
    MU_LAW = "mu_law"


@attr.s(auto_attribs=True)
class CameraQuantizer:
    """
    A camera quantizer that discretizes and undiscretizes a continuous camera input with y (pitch) and x (yaw) components.

    Parameters:
    - camera_binsize: The size of the bins used for quantization. In case of mu-law quantization, it corresponds to the average binsize.
    - camera_maxval: The maximum value of the camera action.
    - quantization_scheme: The quantization scheme to use. Currently, two quantization schemes are supported:
    - Linear quantization (default): Camera actions are split uniformly into discrete bins
    - Mu-law quantization: Transforms the camera action using mu-law encoding (https://en.wikipedia.org/wiki/%CE%9C-law_algorithm)
    followed by the same quantization scheme used by the linear scheme.
    - mu: Mu is the parameter that defines the curvature of the mu-law encoding. Higher values of
    mu will result in a sharper transition near zero. Below are some reference values listed
    for choosing mu given a constant maxval and a desired max_precision value.
    maxval = 10 | max_precision = 0.5  | μ ≈ 2.93826
    maxval = 10 | max_precision = 0.4  | μ ≈ 4.80939
    maxval = 10 | max_precision = 0.25 | μ ≈ 11.4887
    maxval = 20 | max_precision = 0.5  | μ ≈ 2.7
    maxval = 20 | max_precision = 0.4  | μ ≈ 4.39768
    maxval = 20 | max_precision = 0.25 | μ ≈ 10.3194
    maxval = 40 | max_precision = 0.5  | μ ≈ 2.60780
    maxval = 40 | max_precision = 0.4  | μ ≈ 4.21554
    maxval = 40 | max_precision = 0.25 | μ ≈ 9.81152
    """

    camera_maxval: int
    camera_binsize: int
    quantization_scheme: str = attr.ib(
        default=QuantizationScheme.LINEAR,
        validator=attr.validators.in_(
            [QuantizationScheme.LINEAR, QuantizationScheme.MU_LAW]
        ),
    )
    mu: float = attr.ib(default=5)

    def discretize(self, xy):
        xy = np.clip(xy, -self.camera_maxval, self.camera_maxval)

        if self.quantization_scheme == QuantizationScheme.MU_LAW:
            xy = xy / self.camera_maxval
            v_encode = np.sign(xy) * (
                np.log(1.0 + self.mu * np.abs(xy)) / np.log(1.0 + self.mu)
            )
            v_encode *= self.camera_maxval
            xy = v_encode

        # Quantize using linear scheme
        return np.round((xy + self.camera_maxval) / self.camera_binsize).astype(
            np.int64
        )

    def undiscretize(self, xy):
        xy = xy * self.camera_binsize - self.camera_maxval

        if self.quantization_scheme == QuantizationScheme.MU_LAW:
            xy = xy / self.camera_maxval
            v_decode = (
                np.sign(xy) * (1.0 / self.mu) * ((1.0 + self.mu) ** np.abs(xy) - 1.0)
            )
            v_decode *= self.camera_maxval
            xy = v_decode
        return xy


class ActionTransformer:
    """Transforms actions between internal array and minerl env format."""

    # @store_args
    def __init__(
        self,
        camera_maxval=10,
        camera_binsize=2,
        camera_quantization_scheme="linear",
        camera_mu=5,
    ):
        self.quantizer = CameraQuantizer(
            camera_maxval=camera_maxval,
            camera_binsize=camera_binsize,
            quantization_scheme=camera_quantization_scheme,
            mu=camera_mu,
        )

    def camera_zero_bin(self):
        return self.camera_maxval // self.camera_binsize

    def discretize_camera(self, xy):
        return self.quantizer.discretize(xy)

    def undiscretize_camera(self, pq):
        return self.quantizer.undiscretize(pq)

    def item_embed_id_to_name(self, item_id):
        # return mc.MINERL_ITEM_MAP[item_id]
        pass

    def dict_to_numpy(self, acs):
        """
        Env format to policy output format.
        """
        act = {
            "buttons": np.stack([acs.get(k, 0) for k in Buttons.ALL], axis=-1),
            "camera": self.discretize_camera(acs["camera"]),
        }
        if not self.human_spaces:
            act.update(
                {
                    "synthetic_buttons": np.stack(
                        [acs[k] for k in SyntheticButtons.ALL], axis=-1
                    ),
                    "place": self.item_embed_name_to_id(acs["place"]),
                    "equip": self.item_embed_name_to_id(acs["equip"]),
                    "craft": self.item_embed_name_to_id(acs["craft"]),
                }
            )
        return act

    def numpy_to_dict(self, acs):
        """
        Numpy policy output to env-compatible format.
        """
        assert acs["buttons"].shape[-1] == len(Buttons.ALL), (
            f"Mismatched actions: {acs}; expected {len(Buttons.ALL)}:\n(  {Buttons.ALL})"
        )
        out = {name: acs["buttons"][..., i] for (i, name) in enumerate(Buttons.ALL)}

        out["camera"] = self.undiscretize_camera(acs["camera"])

        return out

    def policy2env(self, acs):
        acs = self.numpy_to_dict(acs)
        return acs

    def env2policy(self, acs):
        nbatch = acs["camera"].shape[0]
        dummy = np.zeros((nbatch,))
        out = {
            "camera": self.discretize_camera(acs["camera"]),
            "buttons": np.stack([acs.get(k, dummy) for k in Buttons.ALL], axis=-1),
        }
        return out


class ProtoToTorchActionMapper:
    def __init__(self):
        self._camera_quantizer = CameraQuantizer(
            camera_maxval=10,
            camera_binsize=2,
            quantization_scheme=QuantizationScheme.LINEAR,
        )
        self.action_transformer = ActionTransformer(**ACTION_TRANSFORMER_KWARGS)
        self.action_mapper = CameraHierarchicalMapping(n_camera_bins=11)

    def annotation_action_to_env_action(self, json_action):
        """
        Converts a json action into a MineRL action.
        Returns (minerl_action, is_null_action)
        """
        # This might be slow...
        env_action = NOOP_ACTION.copy()
        # As a safeguard, make camera action again so we do not override anything
        env_action["camera"] = np.array([0, 0])

        is_null_action = True
        keyboard_keys = json_action.keyboard.keys  ## this is a list
        for key in keyboard_keys:
            # You can have keys that we do not use, so just skip them
            # NOTE in original training code, ESC was removed and replaced with
            #      "inventory" action if GUI was open.
            #      Not doing it here, as BASALT uses ESC to quit the game.
            if key in KEYBOARD_BUTTON_MAPPING:
                env_action[KEYBOARD_BUTTON_MAPPING[key]] = 1
                is_null_action = False
            else:
                if key in IGNORED_KEYS:
                    continue
                else:
                    print(f"{key} not in the mapping")

        mouse = json_action.mouse
        camera_action = env_action["camera"]
        if mouse.dy:
            is_null_action = False
            camera_action[0] = mouse.dy * CAMERA_SCALER
        else:
            camera_action[0] = 0.0

        if mouse.dx:
            is_null_action = False
            camera_action[1] = mouse.dx * CAMERA_SCALER
        else:
            camera_action[1] = 0.0

        if abs(camera_action[0]) > 180:
            camera_action[0] = 0
        if abs(camera_action[1]) > 180:
            camera_action[1] = 0

        if mouse.buttons:
            mouse_buttons = mouse.buttons
            if 0 in mouse_buttons:
                env_action["attack"] = 1
                is_null_action = False
            if 2 in mouse_buttons:
                env_action["use"] = 1
                is_null_action = False
            ## @yuguang I don't know which is pickItem button..
            # if 2 in mouse_buttons:
            #     env_action["pickItem"] = 1
            #     is_null_action = False
        return env_action, is_null_action

    def to_policy_action(
        self, action_proto: video_annotation_pb2.LowLevelAction
    ) -> FactoredActionType:
        minerl_action_transformed, is_null_action = (
            self.annotation_action_to_env_action(action_proto)
        )
        minerl_action = self.action_transformer.env2policy(minerl_action_transformed)
        if minerl_action["camera"].ndim == 1:
            minerl_action = {k: v[None] for k, v in minerl_action.items()}
        action = self.action_mapper.from_factored(minerl_action)
        return action


def batch_recursive_objects(ls, check_shape: bool = False):
    """Batch a list of objects into one object so the lowest level arrays are batched
    and everything else has the same structure.

    All objects in the list must have the same structure.
    Concat along the already existing batch dimension.

    Simple example:
    >>> a = np.random.rand(1, 1, 4)
    >>> b = np.random.rand(2, 2, 4)
    >>> c = [{'a': a, 'b': b}, {'a': a, 'b': b}]
    >>> print_recursive_shape('c', c)
    c -> list(2)
    c[0] -> dict(2)
    c[0].a = (1, 1, 4)
    c[0].b = (2, 2, 4)
    c[1] -> dict(2)
    c[1].a = (1, 1, 4)
    c[1].b = (2, 2, 4)

    >>> print_recursive_shape('batch_recursive_objects(c)', batch_recursive_objects(c))
    batch_recursive_objects(c) -> dict(2)
    batch_recursive_objects(c).a = (2, 1, 4)
    batch_recursive_objects(c).b = (4, 2, 4)

    Complicated example:
    >>> a = np.random.rand(1, 1, 4)
    >>> b = np.random.rand(2, 2, 4)
    >>> c = {'a': a, 'b': b, 't': (a, b), 'n': None}
    >>> d = {'a': a, 'b': b, 't': (a, b), 'n': None}
    >>> e = [c, d]
    >>> print_recursive_shape('e', e)
    e -> list(2)
    e[0] -> dict(4)
    e[0].a = (1, 1, 4)
    e[0].b = (2, 2, 4)
    e[0].t -> tuple(2)
    e[0].t[0] = (1, 1, 4)
    e[0].t[1] = (2, 2, 4)
    e[0].n -> None
    e[1] -> dict(4)
    e[1].a = (1, 1, 4)
    e[1].b = (2, 2, 4)
    e[1].t -> tuple(2)
    e[1].t[0] = (1, 1, 4)
    e[1].t[1] = (2, 2, 4)
    e[1].n -> None

    >>> print_recursive_shape('batch_recursive_objects(e)', batch_recursive_objects(e))
    batch_recursive_objects(e) -> dict(4)
    batch_recursive_objects(e).a = (2, 1, 4)
    batch_recursive_objects(e).b = (4, 2, 4)
    batch_recursive_objects(e).t -> tuple(2)
    batch_recursive_objects(e).t[0] = (2, 1, 4)
    batch_recursive_objects(e).t[1] = (4, 2, 4)
    batch_recursive_objects(e).n -> None
    """
    first = ls[0]
    if isinstance(first, dict):
        # Explanation for the below line:
        #   - ls[0] is a dict, so we can iterate over its keys
        #   - for each key, we get a list of values from each dict in ls
        #   - we batch the list of values
        #   - we return a dict with the same keys as ls[0] and the batched values
        return {k: batch_recursive_objects([d[k] for d in ls]) for k in first}
    elif isinstance(first, list):
        # Similar to the above, but for lists
        return [batch_recursive_objects([l[i] for l in ls]) for i in range(len(first))]
    elif isinstance(first, tuple) and hasattr(first, "_fields"):  # Check if NamedTuple
        # Convert each NamedTuple to a dict with field names as keys
        return type(first)(
            **{
                field: batch_recursive_objects([getattr(l, field) for l in ls])
                for field in first._fields
            }
        )
    elif isinstance(first, tuple):
        # Similar to the above, but for tuples
        return tuple(
            batch_recursive_objects([l[i] for l in ls]) for i in range(len(first))
        )
    elif first is None:
        return None
    else:
        if check_shape:  # might be slow
            assert all([e.shape == first.shape for e in ls]), (
                "All objects must have the same shape"
            )

        if isinstance(first, np.ndarray):
            return np.concatenate(ls, axis=0)
        elif isinstance(first, th.Tensor):
            return th.cat(ls, dim=0)
        else:
            print(9)
            raise ValueError(
                f"Unsupported type: {type(first)}."
                "Only numpy arrays and torch tensors are supported "
                "for non-(dict, list, tuple, None) objects"
            )
