# Source this script into your shell.

export ELEFANT_ML_PLAYGROUND_ROOT=`git rev-parse --show-toplevel`

cd "$ELEFANT_ML_PLAYGROUND_ROOT"

conda activate steve
export PYTHONPATH="$ELEFANT_ML_PLAYGROUND_ROOT:$ELEFANT_ML_PLAYGROUND_ROOT/VPT/Video-Pre-Training"
