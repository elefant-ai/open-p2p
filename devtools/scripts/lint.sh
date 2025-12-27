#! /bin/bash

# Exit if any command fails
set -e

REPO_ROOT=$(git rev-parse --show-toplevel)
cd $REPO_ROOT

# Run black to format python. IF CI=1 then run in check mode.
echo "Linting python ruff"
if [ "$CI" = "1" ]; then
  uv run ruff format --check --exclude "*.ipynb" --exclude ".git/*"
  cargo fmt --all --manifest-path elefant_rust/Cargo.toml -- --check
  cargo clippy --all-targets --all-features --manifest-path elefant_rust/Cargo.toml -- -D warnings
else
  uv run ruff format --exclude "*.ipynb" --exclude ".git/*"
  cargo fmt --all --manifest-path elefant_rust/Cargo.toml
  cargo clippy --all-targets --all-features --manifest-path elefant_rust/Cargo.toml -- -D warnings
fi
