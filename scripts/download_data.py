import argparse
import shutil
import tarfile
from pathlib import Path
from huggingface_hub import snapshot_download

def prepare_for_training(toy: bool, output: str):
    if not toy:
        output_path = Path(output)
        
        tar_files = list(output_path.rglob("*.tar.gz"))
        print(f"Found tar files: {tar_files}")
        
        if not tar_files:
            print("No tar files found!")
            return

        moved_tar_files = []
        for tar_file in tar_files:
            dest = output_path.parent / tar_file.name
            print(f"Moving {tar_file} -> {dest}")
            shutil.move(str(tar_file), str(dest))
            moved_tar_files.append(dest)

        shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        for tar_file in sorted(moved_tar_files):
            print(f"Extracting {tar_file.name} to {output_path}...")
            with tarfile.open(tar_file, "r:gz") as tar:
                tar.extractall(path=output_path)
            tar_file.unlink()
            print(f"Deleted {tar_file.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--toy", action="store_true", help="Download toy examples instead of full data")
    parser.add_argument("--start", type=int, default=1, help="Start batch index (full data only)")
    parser.add_argument("--end", type=int, help="End batch index inclusive (full data only)")
    parser.add_argument("--output", type=str, default="dataset", help="Output directory")
    args = parser.parse_args()

    repo = "elefantai/p2p-toy-examples" if args.toy else "elefantai/p2p-full-data"
    patterns = None if args.toy or args.end is None else [f"*batch_{i:05d}*" for i in range(args.start, args.end + 1)]

    snapshot_download(repo, repo_type="dataset", allow_patterns=patterns, local_dir=args.output)
    prepare_for_training(args.toy, args.output)

if __name__ == "__main__":
    main()
