import pydantic_yaml
from pydantic import BaseModel, ConfigDict
from typing import Type
import logging
from string import Template
import re
import os
import fsspec


class ConfigBase(BaseModel):
    model_config = ConfigDict(extra="ignore")


def _substitute_env_vars(content: str) -> str:
    """Replace ${VAR} or $VAR in string with environment variable values."""

    # Hack to deal with different laptop usernames.
    env_vars = os.environ.copy()
    if "USER" in env_vars and env_vars["USER"] == "j":
        env_vars["USER"] = "jj"

    # Handle ${VAR} style
    pattern = re.compile(r"\${([^}^{]+)}")
    content = pattern.sub(lambda m: env_vars.get(m.group(1), ""), content)

    # Handle $VAR style
    return Template(content).safe_substitute(env_vars)


def load_config(path: str, model: Type[BaseModel]) -> BaseModel:
    """Load a config from a yaml file and return a pydantic model."""
    if path is None:
        # If no path is provided, return a default model.
        content = None
    else:
        with fsspec.open(path, "r") as f:
            content = f.read().strip()

    if content:
        # Substitute environment variables before parsing
        content = _substitute_env_vars(content)

    # If the yaml file is empty, return a default model.
    if not content:
        m = model()
    else:
        m = pydantic_yaml.parse_yaml_raw_as(model, content)
    logging.info(f"Loaded config from {path}: {m}")
    return m
