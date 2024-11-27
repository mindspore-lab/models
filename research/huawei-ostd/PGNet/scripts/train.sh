CONFIG=$1
CARDS=$2

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
mpirun \
    --allow-run-as-root \
    -n $CARDS \
    python tools/train.py \
    --config $CONFIG
    ${@:3}
