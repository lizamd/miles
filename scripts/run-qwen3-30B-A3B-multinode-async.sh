#!/bin/bash
# Async, non-colocate multinode launcher for Qwen3-30B-A3B.
#
# Node layout (default):
#   - Node 0             -> training  (TRAIN_NUM_NODES=1)
#   - Nodes 1..N-1       -> rollout   (ROLLOUT_NUM_NODES = ACTOR_NUM_NODES - TRAIN_NUM_NODES)
#
# Weight sync is NCCL broadcast (miles.UpdateWeightFromDistributed). On this cluster NCCL
# will use RDMA/IB automatically once NCCL_IB_HCA and NCCL_SOCKET_IFNAME are set (both
# auto-detected below, overridable via env).
#
# Batch size scales linearly with ROLLOUT_NUM_NODES unless the user overrides it:
#   ROLLOUT_BATCH_SIZE = 32  * ROLLOUT_NUM_NODES
#   GLOBAL_BATCH_SIZE  = 256 * ROLLOUT_NUM_NODES

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python
pkill -9 redis

# ── SGLang rollout-side runtime fixes for async MoE weight sync ───────── #
python3 <<'PY'
import pathlib

SENTINEL = "MILES_RUNTIME_FIX_V2"


def prepare_source(path_str: str):
    path = pathlib.Path(path_str)
    if not path.exists():
        print(f"[miles-runtime-fix] {path} not found; skipping")
        return None, None

    live_src = path.read_text()
    orig = path.with_suffix(path.suffix + ".miles_orig")
    if orig.exists():
        src = orig.read_text()
    else:
        if SENTINEL in live_src:
            print(f"[miles-runtime-fix] {path.name} already patched")
            return None, None
        if "MILES_DIAG" in live_src:
            raise RuntimeError(
                f"{path} still contains a prior diagnostic patch but no backup snapshot was found"
            )
        orig.write_text(live_src)
        src = live_src
        print(f"[miles-runtime-fix] snapshotted original -> {orig}")

    return path, src


def patch_qwen3_moe():
    path, src = prepare_source("/sgl-workspace/sglang/python/sglang/srt/models/qwen3_moe.py")
    if path is None:
        return

    old = (
        "        # Cache params_dict to avoid repeated expensive traversal of model parameters\n"
        "        if not hasattr(self, \"_cached_params_dict\"):\n"
        "            self._cached_params_dict = dict(self.named_parameters())\n"
        "        params_dict = self._cached_params_dict\n"
    )
    new = (
        "        # MILES_RUNTIME_FIX_V1: post-load hooks may rebind Parameters, so rebuild\n"
        "        # the lookup table for every load_weights call.\n"
        "        params_dict = dict(self.named_parameters())\n"
    )
    if old not in src:
        raise RuntimeError(f"{path}: expected qwen3_moe load_weights anchor not found")

    path.write_text(src.replace(old, new, 1))
    print(f"[miles-runtime-fix] patched {path.name}")


def patch_unquant_moe():
    path, src = prepare_source("/sgl-workspace/sglang/python/sglang/srt/layers/quantization/unquant.py")
    if path is None:
        return

    old = (
        "        if _should_use_aiter_moe:\n"
        "            layer.w13_weight = torch.nn.Parameter(\n"
        "                shuffle_weight(layer.w13_weight.data, (16, 16)),\n"
        "                requires_grad=False,\n"
        "            )\n"
        "            torch.cuda.empty_cache()\n"
        "            layer.w2_weight = torch.nn.Parameter(\n"
        "                shuffle_weight(layer.w2_weight.data, (16, 16)),\n"
        "                requires_grad=False,\n"
        "            )\n"
        "            torch.cuda.empty_cache()\n"
    )
    new = (
        "        if _should_use_aiter_moe:\n"
        "            # MILES_RUNTIME_FIX_V2: shuffle in place so CUDA graphs captured at\n"
        "            # init stay valid across update_weights_from_distributed. Rebinding\n"
        "            # the Parameter + empty_cache() frees the old GPU storage that init\n"
        "            # CUDA graphs reference, causing a ROCm memory access fault on the\n"
        "            # first post-update decode replay.\n"
        "            _shuffled_w13 = shuffle_weight(layer.w13_weight.data, (16, 16))\n"
        "            layer.w13_weight.data.copy_(_shuffled_w13)\n"
        "            layer.w13_weight.is_shuffled = True\n"
        "            del _shuffled_w13\n"
        "            torch.cuda.empty_cache()\n"
        "            _shuffled_w2 = shuffle_weight(layer.w2_weight.data, (16, 16))\n"
        "            layer.w2_weight.data.copy_(_shuffled_w2)\n"
        "            layer.w2_weight.is_shuffled = True\n"
        "            del _shuffled_w2\n"
        "            torch.cuda.empty_cache()\n"
    )
    if old not in src:
        raise RuntimeError(f"{path}: expected unquant MoE shuffle anchor not found")

    path.write_text(src.replace(old, new, 1))
    print(f"[miles-runtime-fix] patched {path.name}")


def patch_weight_checker():
    path, src = prepare_source("/sgl-workspace/sglang/python/sglang/srt/utils/weight_checker.py")
    if path is None:
        return

    old = (
        "    def _model_state(self):\n"
        "        # TODO: support EAGLE etc (e.g. yield from both main model and draft model)\n"
        "        yield from self._model_runner.model.named_parameters()\n"
        "        yield from self._model_runner.model.named_buffers()\n"
    )
    new = (
        "    def _model_state(self):\n"
        "        # TODO: support EAGLE etc (e.g. yield from both main model and draft model)\n"
        "        yield from self._model_runner.model.named_parameters()\n"
        "        yield from self._named_persistent_buffers()\n"
        "\n"
        "    def _named_persistent_buffers(self):\n"
        "        # MILES_RUNTIME_FIX_V1: derived non-persistent buffers are regenerated,\n"
        "        # not restored by online weight sync.\n"
        "        for module_name, module in self._model_runner.model.named_modules():\n"
        "            non_persistent = getattr(module, \"_non_persistent_buffers_set\", set())\n"
        "            for buffer_name, buffer in module._buffers.items():\n"
        "                if buffer is None or buffer_name in non_persistent:\n"
        "                    continue\n"
        "                full_name = f\"{module_name}.{buffer_name}\" if module_name else buffer_name\n"
        "                yield full_name, buffer\n"
    )
    if old not in src:
        raise RuntimeError(f"{path}: expected weight_checker anchor not found")

    path.write_text(src.replace(old, new, 1))
    print(f"[miles-runtime-fix] patched {path.name}")


patch_qwen3_moe()
patch_unquant_moe()
patch_weight_checker()
PY
patch_status=$?
if [ "${patch_status}" -ne 0 ]; then echo "WARN: SGLang patches did not apply (likely version mismatch); continuing"; fi
if false; then
    echo "Failed to patch SGLang runtime fixes"
    exit "${patch_status}"
fi

set -euxo pipefail

# ==================== Platform Detection ====================
if [ -e /dev/kfd ] || python3 -c "import torch; assert torch.version.hip" 2>/dev/null; then
    GPU_VENDOR="amd"
elif command -v nvidia-smi &>/dev/null; then
    GPU_VENDOR="nvidia"
else
    echo "ERROR: No supported GPU detected (need NVIDIA or AMD)"
    exit 1
fi
echo "Detected GPU vendor: ${GPU_VENDOR}"

# ==================== Configurable Paths ====================
MODEL_DIR="${MODEL_DIR:-/root}"
DATA_DIR="${DATA_DIR:-/root}"
export MODEL_DIR DATA_DIR

# ==================== Platform-Specific Setup ====================
if [ "$GPU_VENDOR" = "amd" ]; then
    export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=${RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES:-"1"}
    export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}
    if [ -z "${HIP_VISIBLE_DEVICES}" ]; then
        NUM_GPUS=0
    else
        NUM_GPUS=$(echo "${HIP_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
    fi
    HAS_NVLINK=0
else
    NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
    if [ "$NVLINK_COUNT" -gt 0 ]; then
        HAS_NVLINK=1
    else
        HAS_NVLINK=0
    fi
    echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
fi

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
TRAIN_WORKDIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-30B-A3B.sh"

# Checkpoint behavior:
# - By default, resume from the previous MILES checkpoint directory.
# - Set RESET_LOAD=1 to ignore stale/incompatible MILES checkpoints and start from torch_dist.
LOAD_DIR_DEFAULT="${MODEL_DIR}/Qwen3-30B-A3B_miles/"
if [ "${RESET_LOAD:-0}" = "1" ]; then
   LOAD_DIR_DEFAULT="${MODEL_DIR}/Qwen3-30B-A3B_torch_dist"
fi
LOAD_DIR="${LOAD_DIR:-${LOAD_DIR_DEFAULT}}"
SAVE_DIR="${SAVE_DIR:-${MODEL_DIR}/Qwen3-30B-A3B_miles/}"

echo "LOAD_DIR=${LOAD_DIR}"
echo "SAVE_DIR=${SAVE_DIR}"

# ==================== Cluster Topology: 1 train + (N-1) rollout ====================
# Default: 2 nodes (1 train + 1 rollout) for the g05+g06 allocation.
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-2}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-${NUM_GPUS}}"
TRAIN_NUM_NODES="${TRAIN_NUM_NODES:-1}"
ROLLOUT_NUM_NODES=$((ACTOR_NUM_NODES - TRAIN_NUM_NODES))
if [ "${ROLLOUT_NUM_NODES}" -lt 1 ]; then
    echo "ERROR: ROLLOUT_NUM_NODES=${ROLLOUT_NUM_NODES} (need ACTOR_NUM_NODES > TRAIN_NUM_NODES)"
    exit 1
fi
ROLLOUT_NUM_GPUS=$((ROLLOUT_NUM_NODES * NUM_GPUS))

# Async batch scaling. Empty => scale linearly with rollout-node count.
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-}"
if [ -z "${ROLLOUT_BATCH_SIZE}" ]; then
    ROLLOUT_BATCH_SIZE=$((32 * ROLLOUT_NUM_NODES))
fi
if [ -z "${GLOBAL_BATCH_SIZE}" ]; then
    GLOBAL_BATCH_SIZE=$((256 * ROLLOUT_NUM_NODES))
fi
UPDATE_WEIGHTS_INTERVAL="${UPDATE_WEIGHTS_INTERVAL:-2}"

echo "ACTOR_NUM_NODES=${ACTOR_NUM_NODES} TRAIN_NUM_NODES=${TRAIN_NUM_NODES} ROLLOUT_NUM_NODES=${ROLLOUT_NUM_NODES}"
echo "ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS} ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE} GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
echo "UPDATE_WEIGHTS_INTERVAL=${UPDATE_WEIGHTS_INTERVAL}"

# Guard: actor-num-nodes passed to miles is TRAIN_NUM_NODES (training-only), NOT ACTOR_NUM_NODES
# (which in this script is the total slurm node count for readability).

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3-30B-A3B
   #--hf-checkpoint ${MODEL_DIR}/Qwen3-30B-A3B-FP8
   --ref-load ${MODEL_DIR}/Qwen3-30B-A3B_torch_dist
   --load ${LOAD_DIR}
   --save ${SAVE_DIR}
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data ${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 3000
   --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1
   --log-passrate

   --global-batch-size ${GLOBAL_BATCH_SIZE}
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime ${DATA_DIR}/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 1
   --skip-eval-before-train
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 20480
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project miles-dev
   --wandb-group qwen3-30B-A3B-multinode-async
   --wandb-team liz-li-amd
   --wandb-key ${WANDB_KEY:-}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.8
   --sglang-max-running-requests 256
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
   --sglang-ep-size 2
   --sglang-enable-dp-attention
   --sglang-dp-size 2
)

# AMD: disable custom all-reduce
if [ "$GPU_VENDOR" = "amd" ]; then
    SGLANG_ARGS+=(--sglang-disable-custom-all-reduce)
   #  SGLANG_ARGS+=(--sglang-attention-backend triton)
fi

export NVTE_USE_CUTLASS_GROUPED_GEMM=1
export NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK=1

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   # MI355X (gfx950): TE fused attn rejects THD packed; flash works.
   --attention-backend flash
   # use TE CK grouped GEMM for MoE
   --moe-grouped-gemm
)

ASYNC_ARGS=(
   --update-weights-interval ${UPDATE_WEIGHTS_INTERVAL}
   --check-weight-update-equal
)

# ==================== Multinode Ray Setup ====================
RAY_NODE_ROLE="${RAY_NODE_ROLE:-head}"
RAY_HEAD_PORT="${RAY_HEAD_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8285}"
RAY_WAIT_FOR_NODES_TIMEOUT_S="${RAY_WAIT_FOR_NODES_TIMEOUT_S:-600}"
RAY_WAIT_INTERVAL_S="${RAY_WAIT_INTERVAL_S:-5}"

DETECTED_LOCAL_NODE_IP=$(hostname -I 2>/dev/null | awk '{for (i=1; i<=NF; i++) if ($i !~ /^127\./) {print $i; exit}}')
LOCAL_NODE_IP="${RAY_NODE_IP:-${DETECTED_LOCAL_NODE_IP}}"
if [ -z "${LOCAL_NODE_IP}" ]; then
    LOCAL_NODE_IP="127.0.0.1"
fi

USER_PROVIDED_MASTER_ADDR=1
if [ -z "${MASTER_ADDR:-}" ] || [ "${MASTER_ADDR}" = "127.0.0.1" ] || [ "${MASTER_ADDR}" = "localhost" ]; then
    USER_PROVIDED_MASTER_ADDR=0
    export MASTER_ADDR="${LOCAL_NODE_IP}"
else
    export MASTER_ADDR
fi

export no_proxy="localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}"
export NO_PROXY="${no_proxy}"
echo "MASTER_ADDR=${MASTER_ADDR} LOCAL_NODE_IP=${LOCAL_NODE_IP}"
echo "RAY_NODE_ROLE=${RAY_NODE_ROLE} RAY_HEAD_PORT=${RAY_HEAD_PORT} RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT}"

if [ "${RAY_NODE_ROLE}" != "head" ] && [ "${RAY_NODE_ROLE}" != "worker" ]; then
    echo "Invalid RAY_NODE_ROLE=${RAY_NODE_ROLE}. Use head or worker."
    exit 1
fi

if [ "${ACTOR_NUM_NODES}" -gt 1 ] && [ "${RAY_NODE_ROLE}" = "worker" ] && [ "${USER_PROVIDED_MASTER_ADDR}" = "0" ]; then
    echo "In multi-node worker mode, set MASTER_ADDR to the Ray head node IP."
    exit 1
fi

# ==================== RDMA / NCCL Env (auto-detected, overridable) ====================
# Detection runs with set +e so a missing `ip` / empty /sys/class/infiniband
# (sandbox containers) does not abort the whole launch under pipefail+e.
set +e
# Socket iface is the NIC that carries MASTER_ADDR (== Ray bootstrap iface).
if [ -z "${NCCL_SOCKET_IFNAME:-}" ]; then
    NCCL_SOCKET_IFNAME=$(ip -o -4 addr show 2>/dev/null | awk -v ip="${LOCAL_NODE_IP}" '$4 ~ ip"/" {print $2; exit}')
    if [ -z "${NCCL_SOCKET_IFNAME}" ]; then
        NCCL_SOCKET_IFNAME=$(ip -o -4 route show to default 2>/dev/null | awk '{print $5; exit}')
    fi
fi
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export NCCL_SOCKET_IFNAME
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${NCCL_SOCKET_IFNAME}}"

# IB HCAs: keep only HCAs whose backing netdev is rdma* (the GPU-side fast fabric).
# Skips HCAs bound to ethX, which are management NICs and would steer NCCL traffic
# onto the bootstrap iface instead of the RoCE fabric.
if [ -z "${NCCL_IB_HCA:-}" ]; then
    _ib_list=""
    if [ -d /sys/class/infiniband ]; then
        for dev in /sys/class/infiniband/*; do
            [ -d "$dev" ] || continue
            ndev=$(cat "$dev/ports/1/gid_attrs/ndevs/0" 2>/dev/null)
            case "$ndev" in
                rdma*)
                    name=$(basename "$dev")
                    if [ -z "$_ib_list" ]; then
                        _ib_list="$name"
                    else
                        _ib_list="${_ib_list},${name}"
                    fi
                    ;;
            esac
        done
    fi
    NCCL_IB_HCA="$_ib_list"
    unset _ib_list
fi
export NCCL_IB_HCA="${NCCL_IB_HCA:-}"
set -e
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
export NCCL_IB_TC="${NCCL_IB_TC:-160}"
export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-22}"
export NCCL_IB_RETRY_CNT="${NCCL_IB_RETRY_CNT:-7}"
export NCCL_IB_QPS_PER_CONNECTION="${NCCL_IB_QPS_PER_CONNECTION:-8}"
export NCCL_PXN_DISABLE="${NCCL_PXN_DISABLE:-0}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-VERSION}"

echo "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} NCCL_IB_HCA=${NCCL_IB_HCA:-<none>}"

# ── Start Ray ────────────────────────────────────────────────────────────── #
ray stop --force 2>/dev/null || true

if [ "${RAY_NODE_ROLE}" = "head" ]; then
    ray start --head \
        --node-ip-address "${MASTER_ADDR}" \
        --port "${RAY_HEAD_PORT}" \
        --num-gpus "${NUM_GPUS}" \
        --disable-usage-stats \
        --dashboard-host=0.0.0.0 \
        --dashboard-port="${RAY_DASHBOARD_PORT}"
else
    ray start \
        --address "${MASTER_ADDR}:${RAY_HEAD_PORT}" \
        --node-ip-address "${LOCAL_NODE_IP}" \
        --num-gpus "${NUM_GPUS}" \
        --disable-usage-stats

    echo "Ray worker started on ${LOCAL_NODE_IP}, joined ${MASTER_ADDR}:${RAY_HEAD_PORT}."
    echo "Waiting for head node to finish training..."
    # Keep the worker process alive so Ray stays up.
    # The launcher (or user) sends SIGTERM/SIGINT to tear down.
    tail -f /dev/null
fi

# ── Head: wait for all workers to join ───────────────────────────────────── #
if [ "${ACTOR_NUM_NODES}" -gt 1 ]; then
    CLUSTER_READY=0
    WAIT_DEADLINE=$((SECONDS + RAY_WAIT_FOR_NODES_TIMEOUT_S))
    while [ "${SECONDS}" -lt "${WAIT_DEADLINE}" ]; do
        ALIVE_NODES=$(python3 - <<'PY' 2>/dev/null || echo 0
import ray
ray.init(address="auto", ignore_reinit_error=True, logging_level=40)
print(sum(1 for n in ray.nodes() if n.get("Alive")))
ray.shutdown()
PY
)
        if [ "${ALIVE_NODES}" -ge "${ACTOR_NUM_NODES}" ]; then
            CLUSTER_READY=1
            break
        fi
        echo "Waiting for Ray cluster: ${ALIVE_NODES}/${ACTOR_NUM_NODES} nodes alive"
        sleep "${RAY_WAIT_INTERVAL_S}"
    done
    if [ "${CLUSTER_READY}" != "1" ]; then
        echo "Timed out waiting for Ray workers. Expected ${ACTOR_NUM_NODES} nodes."
        exit 1
    fi
    echo "Ray cluster ready: ${ALIVE_NODES}/${ACTOR_NUM_NODES} nodes."
fi

# ── Head: submit async training job ──────────────────────────────────────── #
MEGATRON_LM_PATH=$(python3 -c "import megatron; import os; print(os.path.dirname(os.path.dirname(megatron.__file__)))" 2>/dev/null || echo "/app/Megatron-LM")

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_LM_PATH}/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"SGLANG_SET_CPU_AFFINITY\": \"0\",
    \"NVTE_USE_CUTLASS_GROUPED_GEMM\": \"1\",
    \"NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK\": \"1\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"no_proxy\": \"${no_proxy}\",
    \"NO_PROXY\": \"${NO_PROXY}\",
    \"NCCL_SOCKET_IFNAME\": \"${NCCL_SOCKET_IFNAME}\",
    \"GLOO_SOCKET_IFNAME\": \"${GLOO_SOCKET_IFNAME}\",
    \"NCCL_IB_HCA\": \"${NCCL_IB_HCA}\",
    \"NCCL_IB_GID_INDEX\": \"${NCCL_IB_GID_INDEX}\",
    \"NCCL_IB_TC\": \"${NCCL_IB_TC}\",
    \"NCCL_IB_TIMEOUT\": \"${NCCL_IB_TIMEOUT}\",
    \"NCCL_IB_RETRY_CNT\": \"${NCCL_IB_RETRY_CNT}\",
    \"NCCL_IB_QPS_PER_CONNECTION\": \"${NCCL_IB_QPS_PER_CONNECTION}\",
    \"NCCL_PXN_DISABLE\": \"${NCCL_PXN_DISABLE}\",
    \"NCCL_NET_GDR_LEVEL\": \"${NCCL_NET_GDR_LEVEL}\",
    \"NCCL_DEBUG\": \"${NCCL_DEBUG}\"
  }
}"

ray job submit --address="http://${MASTER_ADDR}:${RAY_DASHBOARD_PORT}" \
   --working-dir "${TRAIN_WORKDIR}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
   --actor-num-nodes "${TRAIN_NUM_NODES}" \
   --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
   --rollout-num-gpus "${ROLLOUT_NUM_GPUS}" \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${ASYNC_ARGS[@]}
