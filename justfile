
set shell := ["zsh", "-cu"]

ELEFANT_ML_PLAYGROUND_ROOT := `git rev-parse --show-toplevel`
ELEFANT_TRUNK_ROOT := ELEFANT_ML_PLAYGROUND_ROOT / "../trunk"


# Default recipe that displays help information
default: help

# list all available recipes
help:
    @just --list

lint:
    ./devtools/scripts/lint.sh

compile_proto:
    protoc \
        "{{ELEFANT_TRUNK_ROOT}}/training_data/proto/video_annotation.proto" \
        "{{ELEFANT_TRUNK_ROOT}}/training_data/proto/video_inference.proto" \
        "{{ELEFANT_TRUNK_ROOT}}/training_data/proto/shared.proto" \
        --proto_path="{{ELEFANT_TRUNK_ROOT}}/training_data/proto" \
        --python_out="{{ELEFANT_ML_PLAYGROUND_ROOT}}/elefant/data/proto"

    uv run python -m grpc_tools.protoc \
        -I "{{ELEFANT_TRUNK_ROOT}}/training_data/proto" \
        --python_out="elefant/data/proto" \
        --pyi_out="elefant/data/proto" \
        --grpc_python_out="elefant/data/proto" \
        "{{ELEFANT_TRUNK_ROOT}}/training_data/proto/video_inference.proto"

    # Fix the stupid imports in grpc python
    uv run protol \
        -o elefant/data/proto \
        --in-place \
        protoc --proto-path="{{ELEFANT_TRUNK_ROOT}}/training_data/proto" video_inference.proto video_annotation.proto shared.proto
    @just lint


# Warning this will delete all local branches except main.
delete-local-branches:
    #! /usr/bin/env zsh
    git branch  | grep -v '\*\|main' | xargs git branch -D

# Run the colab backend.
# Follow instructions here: https://research.google.com/colaboratory/local-runtimes.html
# to connect.
# First set a token with `setec set '/$USER/colab_token'
colab:
    #! /usr/bin/env zsh
    TOKEN=$(setec get "/$USER/colab_token")
    echo "Use url http://localhost:8888/tree?token=$TOKEN"
    uv run jupyter notebook \
        --ServerApp.allow_origin='https://colab.research.google.com' \
        --port=8888 \
        --ServerApp.port_retries=0 \
        --ServerApp.token="$TOKEN" \
        --allow-root

vscode_debug *ARGS:
    uv run python -m debugpy --listen 5678 --wait-for-client {{ARGS}}

kill_stragglers:
    #! /usr/bin/env zsh
    # Kill all processes running uv python.
    # It's ok if this fails, just means it didn't find any stragglers.
    pkill -9 -f "/tmp/elefant-uv-env/bin/python[3]?" || true

clean_tmp_data:
    rm -rf /ephemeral/elefant_tmp_data
    rm -rf /tmp/elefant_zmq

download_youtube:
    uv run --extra youtube python elefant/data/youtube/download_and_upload.py

download_twitch:
    uv run --extra youtube python elefant/data/youtube/twitch/download_and_upload.py

reset:
    just kill_stragglers
    just clean_tmp_data

data-server:
    uv run --extra youtube python elefant/data/data_server.py

godot-eval *ARGS:
    uv run godot-eval {{ARGS}}

run-all-eval *ARGS:
    uv run python elefant/eval/run_eval.py {{ARGS}}

clean_windows:
    echo "--- Cleaning up project-specific temporary files ---"
    rm -rf /tmp/eval
    rm -rf /tmp/elefant*
    rm -rf /tmp/pymp-*
    rm -rf /tmp/torchinductor_$(whoami)
    rm -rf /tmp/pyright-*
    rm /tmp/inference.log
    echo "--- Cleanup complete ---"

clean_torch_compiler:
    echo "--- Cleaning up torch compiler cache ---"
    rm -rf /tmp/torch_compiler