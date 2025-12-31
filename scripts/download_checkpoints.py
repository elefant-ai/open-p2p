import sys
from huggingface_hub import snapshot_download

# Check if size argument is provided
if len(sys.argv) < 2:
    print("Usage: python download.py <150M|300M|600M|1200M>")
    sys.exit(1)

target_size = sys.argv[1]
repo_id = "guaguaa/open-p2p"

print(f"Downloading {target_size} from {repo_id}...")

snapshot_download(
    repo_id=repo_id,
    allow_patterns=f"{target_size}/*",
    local_dir="./checkpoints",
)

print("Done.")
