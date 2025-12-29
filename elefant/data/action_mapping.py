import json
import logging
from typing import List
from copy import deepcopy
import math
import torch
import pydantic
from typing import Optional
from elefant.config import ConfigBase
from elefant.data.proto import shared_pb2
from typing import NamedTuple


class StructuredAction(NamedTuple):
    keys: torch.Tensor
    mouse_buttons: torch.Tensor
    mouse_delta_x: torch.Tensor
    mouse_delta_y: torch.Tensor


class UniversalAutoregressiveActionMappingConfig(ConfigBase):
    max_keys: int = pydantic.Field(
        default=4,
        ge=1,
        le=6,
        description="Maximum number of keys that can be pressed at once",
    )
    max_mouse_keys: int = pydantic.Field(
        default=2,
        ge=1,
        le=3,
        description="Maximum number of mouse keys that can be pressed at once",
    )


UniversalAutogressiveKeyboardEncodingMap = {
    ## keys from minecraft annotations, ignored for now
    "F1": None,
    "F2": None,
    "F3": None,
    "F4": None,
    "F5": None,
    "F6": None,
    "F7": None,
    "F8": None,
    "F9": None,
    "F10": None,
    "F11": None,
    "F12": None,
    "F20": None,
    "PrintScreen": None,
    "Pause": None,
    "Equal": None,
    "Semicolon": None,
    "LeftBracket": None,
    "RightBracket": None,
    "LeftWin": None,
    "End": None,
    "Subtract": None,
    "Menu": None,
    "Period": None,
    "Comma": None,
    "LeftAlt": None,
    "RightAlt": None,
    "LeftControl": None,
    "RightControl": None,
    "Delete": None,
    "Add": None,
    "Decimal": None,
    "Minus": None,
    "PageDown": None,
    "PageUp": None,
    "Home": None,
    "CapsLock": None,
    "Backslash": None,
    "Multiply": None,
    "GraveAccent": None,
    "RightWin": None,
    "ScrollLock": None,
    "NumLock": None,
    "Apostrophe": None,
    "Insert": None,
    "Divide": None,
    # Keys we currently ignore but don't want to warn on.
    "Escape": None,
    "Tab": None,
    "Enter": None,
    "BackQuote": None,
    "Slash": None,
    "Backspace": None,
    "VolumeMute": None,
    "VolumeDown": None,
    "VolumeUp": None,
    "0": None,
    "5": None,
    "6": None,
    "7": None,
    "8": None,
    "9": None,
    "b": None,
    "c": None,
    "g": None,
    "h": None,
    "i": None,
    "j": None,
    "k": None,
    "l": None,
    "m": None,
    "n": None,
    "o": None,
    "p": None,
    "r": None,
    "t": None,
    "u": None,
    "v": None,
    "x": None,
    "y": None,
    # All above here should be unused keys.
    "_no_key": 0,
    "Space": 1,
    # Number keys
    "1": 2,
    "2": 3,
    "3": 4,
    "4": 5,
    "a": 6,
    "d": 7,
    "e": 8,
    "f": 9,
    "q": 10,
    "w": 11,
    "s": 12,
    "z": 13,
    "DownArrow": 14,
    "UpArrow": 15,
    "LeftArrow": 16,
    "RightArrow": 17,
    "LeftShift": 18,
    "RightShift": 19,
}


# Mouse keys are encoded as string numbers.
MouseKeyActionMapping = {
    "_no_button": 0,
    # Mouse keys are numbered.
    # Typically: 0 is left click, 1 is right click, 2 is middle click.
    "0": 1,
    "1": 2,
    "2": 3,
}

# Use: https://colab.research.google.com/drive/1nFZjRIVNH2_EzAa3_GKGWgv3TiyM16UN?usp=sharing
# to visualize / estimate bins.
MOUSE_X_BIN_EDGES = [
    -5.001e02,
    -1.441e02,
    -7.610e01,
    -4.610e01,
    -2.910e01,
    -1.810e01,
    -1.110e01,
    -6.100e00,
    -3.100e00,
    -1.100e00,
    -1.000e-01,
    0.000e00,
    1.000e00,
    3.000e00,
    6.000e00,
    1.100e01,
    1.800e01,
    2.900e01,
    4.600e01,
    7.600e01,
    1.440e02,
    5.000e02,
]

MOUSE_X_BIN_CENTERS = [
    -501,
    -322,
    -110,
    -61,
    -38,
    -24,
    -15,
    -9,
    -5,
    -2,
    -1,
    0,
    1,
    2,
    5,
    9,
    15,
    24,
    38,
    61,
    110,
    322,
    501,
]

MOUSE_X_BIN_CONSERVATIVE_CENTERS = [
    -501,
    -145,
    -77,
    -47,
    -30,
    -19,
    -12,
    -7,
    -4,
    -2,
    -1,
    0,
    1,
    2,
    4,
    7,
    12,
    19,
    30,
    47,
    77,
    145,
    501,
]

MOUSE_Y_BIN_EDGES = [
    -1.501e02,
    -2.410e01,
    -1.210e01,
    -7.100e00,
    -4.100e00,
    -3.100e00,
    -1.100e00,
    -1.000e-01,
    0.000e00,
    1.000e00,
    3.000e00,
    4.000e00,
    7.000e00,
    1.200e01,
    2.400e01,
    1.500e02,
]

MOUSE_Y_BIN_CENTERS = [
    -151,
    -87,
    -18,
    -10,
    -6,
    -4,
    -2,
    -1,
    0,
    1,
    2,
    4,
    6,
    10,
    18,
    87,
    151,
]

MOUSE_Y_BIN_CONSERVATIVE_CENTERS = [
    -151,
    -25,
    -13,
    -8,
    -5,
    -4,
    -2,
    -1,
    0,
    1,
    2,
    4,
    5,
    8,
    13,
    25,
    151,
]

# calculated from https://colab.research.google.com/drive/1nFZjRIVNH2_EzAa3_GKGWgv3TiyM16UN?usp=sharing
MOUSE_X_STD = 96
MOUSE_Y_STD = 22


class UniversalAutoregressiveActionMapping:
    def __init__(
        self, config: Optional[UniversalAutoregressiveActionMappingConfig] = None
    ):
        if config is None:
            config = UniversalAutoregressiveActionMappingConfig()
        self.config = config
        self._setup_reverse_map()

        self._mouse_delta_x_edges = torch.tensor(MOUSE_X_BIN_EDGES, dtype=torch.float32)
        self._mouse_delta_y_edges = torch.tensor(MOUSE_Y_BIN_EDGES, dtype=torch.float32)

    def make_empty_action(self, T: int) -> StructuredAction:
        device = self._mouse_delta_x_edges.device
        return StructuredAction(
            keys=torch.zeros(
                T,
                self.get_number_of_keyboard_actions(),
                dtype=torch.long,
                device=device,
            ),
            mouse_buttons=torch.zeros(
                T,
                self.get_number_of_mouse_button_actions(),
                dtype=torch.long,
                device=device,
            ),
            mouse_delta_x=torch.zeros(T, 1, dtype=torch.long, device=device),
            mouse_delta_y=torch.zeros(T, 1, dtype=torch.long, device=device),
        )

    def reshape_structured_action(
        self, action: StructuredAction, B: int, T: int
    ) -> StructuredAction:
        """
        Reshape the structured action to have shape (B, T, -1).
        This is used instead of pytree because pytree leads to slow inference.
        """
        return StructuredAction(
            keys=action.keys.reshape(B, T, -1),
            mouse_buttons=action.mouse_buttons.reshape(B, T, -1),
            mouse_delta_x=action.mouse_delta_x.reshape(B, T, -1),
            mouse_delta_y=action.mouse_delta_y.reshape(B, T, -1),
        )

    def get_number_of_actions(self) -> StructuredAction:
        """How many actions are there in the autoregressive sequence."""
        return StructuredAction(
            keys=self.get_number_of_keyboard_actions(),
            mouse_buttons=self.get_number_of_mouse_button_actions(),
            mouse_delta_x=1,
            mouse_delta_y=1,
        )

    def get_number_of_action_choices(self) -> StructuredAction:
        return StructuredAction(
            keys=self.get_number_of_keyboard_choices(),
            mouse_buttons=self.get_number_of_mouse_button_choices(),
            mouse_delta_x=self.get_n_mouse_x_bins(),
            mouse_delta_y=self.get_n_mouse_y_bins(),
        )

    def get_n_mouse_y_bins(self) -> int:
        # The final bucket is for items past the far right edge.
        return len(MOUSE_Y_BIN_EDGES) + 1

    def get_n_mouse_x_bins(self) -> int:
        return len(MOUSE_X_BIN_EDGES) + 1

    def get_number_of_keyboard_actions(self) -> int:
        return self.config.max_keys

    def get_number_of_mouse_button_actions(self) -> int:
        return self.config.max_mouse_keys

    def get_number_of_keyboard_choices(self) -> int:
        """Returns the number of actions as an int."""
        number_of_actions = (
            max(
                value
                for value in UniversalAutogressiveKeyboardEncodingMap.values()
                if value is not None
            )
            + 1
        )
        return number_of_actions

    def get_number_of_mouse_button_choices(self) -> int:
        """Returns the number of actions as an int."""
        number_of_mouse_keys = (
            max(v for v in MouseKeyActionMapping.values() if v is not None) + 1
        )
        return number_of_mouse_keys

    def get_seq_len(self) -> int:
        """Returns the length of the autoregressive sequence."""
        return (
            self.config.max_keys + self.config.max_mouse_keys + 2
        )  # +2 for mouse delta x, delta y

    def _setup_reverse_map(self):
        self.keyboard_reverse_map = {}
        self.mouse_button_reverse_map = {}
        for k, v in UniversalAutogressiveKeyboardEncodingMap.items():
            if v is None:
                continue
            if v in self.keyboard_reverse_map:
                if len(k) < len(self.keyboard_reverse_map[v]):
                    logging.warning(
                        f"Overwriting {self.keyboard_reverse_map[v]} with {k} because it's shorter"
                    )
                    self.keyboard_reverse_map[v] = k
            else:
                self.keyboard_reverse_map[v] = k
        for k, v in MouseKeyActionMapping.items():
            if v in self.mouse_button_reverse_map:
                raise ValueError(f"Mouse value {v} is duplicated.")
            self.mouse_button_reverse_map[v] = k

    def action_to_tensor(
        self,
        keys: List[str],
        mouse_buttons: List[str],
        mouse_delta_px: shared_pb2.Vec2Int,
    ) -> StructuredAction:
        # If you change this code, check that the code in Stage3LabelledBCLightning.action_in_to_tokens / action_out_tokens_to_logits is also updated.

        # Keyboard actions
        keys_down = set()
        for k in keys:
            if k in UniversalAutogressiveKeyboardEncodingMap:
                v = UniversalAutogressiveKeyboardEncodingMap[k]
                if v is not None:
                    keys_down.add(v)
            else:
                logging.warning(f"Key {k} not in our keymap.")

        keys_down = sorted(list(keys_down))
        if len(keys_down) > self.get_number_of_keyboard_actions():
            logging.warning(
                f"Truncate {keys_down} to {self.get_number_of_keyboard_actions()}."
            )
            keys_down = keys_down[: self.get_number_of_keyboard_actions()]
        else:
            keys_down = keys_down + [0] * (
                self.get_number_of_keyboard_actions() - len(keys_down)
            )
        keys_down = torch.tensor([keys_down], dtype=torch.int64)

        # Mouse buttons
        mouse_buttons_down = set()
        for b in mouse_buttons:
            if b in MouseKeyActionMapping:
                v = MouseKeyActionMapping[b]
                if v is not None:
                    mouse_buttons_down.add(v)
            else:
                logging.warning(f"Mouse button {b} not in our mouse button map.")
        mouse_buttons_down = sorted(list(mouse_buttons_down))
        if len(mouse_buttons_down) > self.get_number_of_mouse_button_actions():
            mouse_buttons_down = mouse_buttons_down[
                : self.get_number_of_mouse_button_actions()
            ]
        else:
            mouse_buttons_down = mouse_buttons_down + [0] * (
                self.get_number_of_mouse_button_actions() - len(mouse_buttons_down)
            )
        mouse_buttons_down = torch.tensor([mouse_buttons_down], dtype=torch.int64)

        # Mouse delta x
        mouse_delta_x = torch.bucketize(
            mouse_delta_px.x, self._mouse_delta_x_edges
        ).view((1,))
        mouse_delta_y = torch.bucketize(
            mouse_delta_px.y, self._mouse_delta_y_edges
        ).view((1,))

        action_out = StructuredAction(
            keys=keys_down,
            mouse_buttons=mouse_buttons_down,
            mouse_delta_x=mouse_delta_x,
            mouse_delta_y=mouse_delta_y,
        )

        return action_out

    def tensor_to_action(
        self,
        action: StructuredAction,
        mouse_sampling_approach: str = "mean",
    ) -> StructuredAction:
        keys = set()
        assert action.keys.shape == (
            1,
            self.config.max_keys,
        )
        for int_v in action.keys.flatten().tolist():
            # Need to reverse lookup the key from the value in the map
            k_name = self.keyboard_reverse_map[int_v]
            if k_name != "_no_key":
                keys.add(k_name)

        keys = sorted(list(keys))
        mouse_buttons = set()
        assert action.mouse_buttons.shape == (
            1,
            self.config.max_mouse_keys,
        )
        for int_v in action.mouse_buttons.flatten().tolist():
            # Need to reverse lookup the key from the value in the map
            k_name = self.mouse_button_reverse_map[int_v]
            if k_name != "_no_button":
                mouse_buttons.add(k_name)
        mouse_buttons = sorted(list(mouse_buttons))

        if mouse_sampling_approach == "mean":
            mouse_delta_x = MOUSE_X_BIN_CENTERS[action.mouse_delta_x.item()]
            mouse_delta_y = MOUSE_Y_BIN_CENTERS[action.mouse_delta_y.item()]
        elif mouse_sampling_approach == "conservative":
            mouse_delta_x = MOUSE_X_BIN_CONSERVATIVE_CENTERS[
                action.mouse_delta_x.item()
            ]
            mouse_delta_y = MOUSE_Y_BIN_CONSERVATIVE_CENTERS[
                action.mouse_delta_y.item()
            ]
        elif mouse_sampling_approach == "truncated_normal":
            if action.mouse_delta_x.item() == 0:
                mouse_delta_x = MOUSE_X_BIN_EDGES[0] - 1
            elif action.mouse_delta_x.item() == self.get_n_mouse_x_bins() - 1:
                mouse_delta_x = MOUSE_X_BIN_EDGES[-1] + 1
            else:
                lb = MOUSE_X_BIN_EDGES[action.mouse_delta_x.item() - 1]
                ub = MOUSE_X_BIN_EDGES[action.mouse_delta_x.item()]
                lb, ub = obtain_lower_upper_bound(lb, ub)
                if ub <= lb:
                    mouse_delta_x = lb
                else:
                    mouse_delta_x = torch.empty(1)
                    _ = torch.nn.init.trunc_normal_(
                        mouse_delta_x, mean=0, std=MOUSE_X_STD, a=lb, b=ub
                    )
                    mouse_delta_x = mouse_delta_x.item()

            if action.mouse_delta_y.item() == 0:
                mouse_delta_y = MOUSE_Y_BIN_EDGES[0] - 1
            elif action.mouse_delta_y.item() == self.get_n_mouse_y_bins() - 1:
                mouse_delta_y = MOUSE_Y_BIN_EDGES[-1] + 1
            else:
                lb = MOUSE_Y_BIN_EDGES[action.mouse_delta_y.item() - 1]
                ub = MOUSE_Y_BIN_EDGES[action.mouse_delta_y.item()]
                lb, ub = obtain_lower_upper_bound(lb, ub)
                if ub <= lb:
                    mouse_delta_y = lb
                else:
                    mouse_delta_y = torch.empty(1)
                    _ = torch.nn.init.trunc_normal_(
                        mouse_delta_y, mean=0, std=MOUSE_Y_STD, a=lb, b=ub
                    )
                    mouse_delta_y = mouse_delta_y.item()
            mouse_delta_x = round(mouse_delta_x)
            mouse_delta_y = round(mouse_delta_y)
        else:
            raise ValueError(f"Invalid sampling approach: {mouse_sampling_approach}")

        return StructuredAction(
            keys=keys,
            mouse_buttons=mouse_buttons,
            mouse_delta_x=mouse_delta_x,
            mouse_delta_y=mouse_delta_y,
        )

    def serialize(self) -> str:
        """Serialize the action mapping to a string."""

        data = {}
        data["max_keys"] = self.config.max_keys
        data["keys"] = deepcopy(UniversalAutogressiveKeyboardEncodingMap)
        return json.dumps(data)


def obtain_lower_upper_bound(lb, ub):
    # we want to exclude point that close to original point
    if lb < 0 and ub < 0:
        # first make them both integers
        lb += 0.1
        ub += 0.1
        ub -= 1
    elif lb < 0 and ub == 0:
        lb = 0
        ub = 0
    elif lb == 0 and ub > 0:
        lb += 1
    else:
        lb += 1
    return lb, ub


# TODO: this is an adhoc parser for gamepad actions.
# We probably want to have a gamepad action class or something later.
def parse_gamepad_actions(gamepad_actions):
    gamepad_left_stick = gamepad_actions.left_stick
    gamepad_right_stick = gamepad_actions.right_stick
    gamepad_left_trigger = gamepad_actions.left_trigger
    gamepad_right_trigger = gamepad_actions.right_trigger
    gamepad_buttons = gamepad_actions.buttons
    pressed_buttons = [field.name for field, _ in gamepad_buttons.ListFields()]
    left_stick_x = gamepad_left_stick.x
    left_stick_y = gamepad_left_stick.y
    left_stick_pressed = gamepad_left_stick.pressed
    right_stick_x = gamepad_right_stick.x
    right_stick_y = gamepad_right_stick.y
    right_stick_pressed = gamepad_right_stick.pressed
    left_trigger = gamepad_left_trigger
    right_trigger = gamepad_right_trigger
    left_stick = (left_stick_x, left_stick_y, left_stick_pressed)
    right_stick = (right_stick_x, right_stick_y, right_stick_pressed)
    if (
        len(pressed_buttons) == 0
        and not left_stick_pressed
        and not right_stick_pressed
        and left_trigger == 0
        and right_trigger == 0
        and left_stick_x == 0
        and left_stick_y == 0
        and right_stick_x == 0
        and right_stick_y == 0
    ):
        return None
    return {
        "pressed_buttons": pressed_buttons,
        "left_stick": left_stick,
        "right_stick": right_stick,
        "left_trigger": left_trigger,
        "right_trigger": right_trigger,
    }
