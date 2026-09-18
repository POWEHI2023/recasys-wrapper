#!/usr/bin/env bash
# Launch HSTU ranking through npu_wrapper.py and torchrun.
set -o pipefail

WRAPPER_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
HSTU_DIR=$WRAPPER_ROOT/recsys-examples/examples/hstu
RECSYS_DIR=$WRAPPER_ROOT/recsys-examples/examples
MEGATRON_DIR=$WRAPPER_ROOT/Megatron-LM
MINDSPEED_DIR=$WRAPPER_ROOT/MindSpeed

wrapper_args=(--rewrite-cuda)
while [[ $# -gt 0 && $1 == --patch ]]; do
    if [[ $# -lt 2 || ! -f $2 ]]; then
        echo "Usage: bash run_npu_wrapper.sh [--patch FILE]... [-- training-args...]" >&2
        exit 2
    fi
    wrapper_args+=(--patch "$(realpath "$2")")
    shift 2
done
if [[ ${1:-} == -- ]]; then
    shift
fi

config_file=${GIN_CONFIG_FILE:-$HSTU_DIR/training/configs/movielen_ranking.gin}
if [[ ! -f $config_file ]]; then
    echo "Gin config not found: $config_file" >&2
    exit 2
fi
config_file=$(realpath "$config_file")

export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-2}
export NPU_PROFILE=${NPU_PROFILE:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}
export PYTHONPATH=${PYTHONPATH:+${PYTHONPATH}:}${RECSYS_DIR}:${HSTU_DIR}:${MEGATRON_DIR}:${MINDSPEED_DIR}

if [[ ! $WORLD_SIZE =~ ^[1-9][0-9]*$ ]]; then
    echo "WORLD_SIZE must be a positive integer" >&2
    exit 2
fi
CPU_CORES=$(getconf _NPROCESSORS_ONLN) || exit 1
if ((WORLD_SIZE > CPU_CORES)); then
    echo "WORLD_SIZE ($WORLD_SIZE) exceeds online CPU cores ($CPU_CORES)" >&2
    exit 2
fi
CORES_PER_NPU=$((CPU_CORES / WORLD_SIZE))
CPU_AFFINITY_CONF=1
for ((i=0; i<WORLD_SIZE; i++)); do
    start_core=$((i * CORES_PER_NPU))
    end_core=$((start_core + CORES_PER_NPU - 1))
    CPU_AFFINITY_CONF+=",npu${i}:${start_core}-${end_core}"
done
export CPU_AFFINITY_CONF
echo "CPU_AFFINITY_CONF=$CPU_AFFINITY_CONF"

cd "$HSTU_DIR" || exit 1
log_dir=${LOG_DIR:-$WRAPPER_ROOT}
mkdir -p "$log_dir"
log_file=$log_dir/ranking_npu_wrapper_$(date '+%Y%m%d_%H%M%S').log
"${PYTHON_BIN:-python}" "$WRAPPER_ROOT/npu_wrapper.py" "${wrapper_args[@]}" --torchrun \
    --nproc_per_node "$WORLD_SIZE" \
    --master_addr "${MASTER_ADDR:-localhost}" \
    --master_port "${MASTER_PORT:-6000}" \
    -- \
    ./training/pretrain_gr_ranking.py \
    --gin-config-file "$config_file" \
    "$@" 2>&1 | tee "$log_file"
