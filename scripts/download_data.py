from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="guaguaa/p2p-toy-examples",
    repo_type="dataset",
    allow_patterns="dataset/*",
    local_dir=".",
    local_dir_use_symlinks=False,
)
