from elefant.config.load_config import ConfigBase


class WandbConfig(ConfigBase):
    enabled: bool = True
    project: str = "elefant"
    exp_name: str = "policy_model"
    tags: list[str] = []
    run_id: str | None = None
