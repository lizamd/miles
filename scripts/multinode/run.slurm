#!/bin/bash
#SBATCH --job-name=miles-multinode
#SBATCH --nodelist=mia1-p01-g[05-06]
#SBATCH --nodes=2
#SBATCH --partition=amd-aim
#SBATCH --account=amd-aim
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_run_%j.log
#SBATCH --error=logs/train_run_%j.err

set -euo pipefail

CONTAINER="${CONTAINER:-miles-dev}"
TRAIN_SCRIPT="miles/scripts/run-qwen3-30B-A3B-multinode-async.sh"
RESET_LOAD="${RESET_LOAD:-0}"
MODEL_DIR="${MODEL_DIR:-/root}"
DATA_DIR="${DATA_DIR:-/root}"
RAY_HEAD_PORT="${RAY_HEAD_PORT:-6379}"

# 2-node async layout: 1 train node (g05) + 1 rollout node (g06).
TRAIN_NUM_NODES="${TRAIN_NUM_NODES:-1}"
UPDATE_WEIGHTS_INTERVAL="${UPDATE_WEIGHTS_INTERVAL:-2}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-}"

# NCCL transport (RoCE / RDMA). MI355 g05/g06 expose rdma0..rdma7.
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eno0}"
NCCL_IB_HCA="${NCCL_IB_HCA:-rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7}"
NCCL_DEBUG="${NCCL_DEBUG:-VERSION}"

# Wandb (passed through to the inner training script).
WANDB_KEY="${WANDB_KEY:-}"

mkdir -p logs
READY_FILE="$(pwd)/logs/ray_head_${SLURM_JOB_ID}.ready"
rm -f "${READY_FILE}"

srun bash -c '
  set -euo pipefail

  port_is_open() {
    python3 -c "import socket, sys; host=sys.argv[1]; port=int(sys.argv[2]); s=socket.socket(); s.settimeout(1);
try:
    s.connect((host, port))
except OSError:
    sys.exit(1)
else:
    s.close()
    sys.exit(0)" "$1" "$2"
  }

  run_in_container() {
    docker exec '"${CONTAINER}"' bash -c "\
      MASTER_ADDR=${HEAD_IP} \
      RAY_NODE_ROLE=${ROLE} \
      RAY_HEAD_PORT='"${RAY_HEAD_PORT}"' \
      ACTOR_NUM_NODES=${SLURM_NNODES} \
      TRAIN_NUM_NODES='"${TRAIN_NUM_NODES}"' \
      UPDATE_WEIGHTS_INTERVAL='"${UPDATE_WEIGHTS_INTERVAL}"' \
      ROLLOUT_BATCH_SIZE='"${ROLLOUT_BATCH_SIZE}"' \
      GLOBAL_BATCH_SIZE='"${GLOBAL_BATCH_SIZE}"' \
      RESET_LOAD='"${RESET_LOAD}"' \
      MODEL_DIR='"${MODEL_DIR}"' \
      DATA_DIR='"${DATA_DIR}"' \
      NCCL_SOCKET_IFNAME='"${NCCL_SOCKET_IFNAME}"' \
      NCCL_IB_HCA='"${NCCL_IB_HCA}"' \
      NCCL_DEBUG='"${NCCL_DEBUG}"' \
      WANDB_KEY='"${WANDB_KEY}"' \
      bash '"${TRAIN_SCRIPT}"'"
  }

  ensure_clean_container() {
    if ! docker container inspect '"${CONTAINER}"' >/dev/null 2>&1; then
      echo "Container '"${CONTAINER}"' not found on $(hostname). Run setup.slurm first."
      exit 1
    fi
    if [ "$(docker inspect -f "{{.State.Running}}" '"${CONTAINER}"')" = "true" ]; then
      docker restart '"${CONTAINER}"' >/dev/null
    else
      docker start '"${CONTAINER}"' >/dev/null
    fi
    sleep 2
  }

  HEAD_HOSTNAME=$(scontrol show hostnames $SLURM_NODELIST | head -1)
  HEAD_IP=$(getent hosts "$HEAD_HOSTNAME" | awk "{print \$1}" | head -1)

  if [ "$SLURM_NODEID" -eq 0 ]; then
    ROLE=head
  else
    ROLE=worker
  fi

  echo "[$(hostname)] role=${ROLE} head=${HEAD_IP} node=${SLURM_NODEID}/${SLURM_NNODES}"

  ensure_clean_container

  if [ "${ROLE}" = "head" ]; then
    run_in_container &
    HEAD_PID=$!

    DEADLINE=$((SECONDS + 60))
    while [ "${SECONDS}" -lt "${DEADLINE}" ] && port_is_open "${HEAD_IP}" '"${RAY_HEAD_PORT}"'; do
      sleep 1
    done

    DEADLINE=$((SECONDS + 180))
    until port_is_open "${HEAD_IP}" '"${RAY_HEAD_PORT}"'; do
      if [ "${SECONDS}" -ge "${DEADLINE}" ]; then
        echo "Timed out waiting for Ray head on ${HEAD_IP}:'"${RAY_HEAD_PORT}"'"
        kill "${HEAD_PID}" 2>/dev/null || true
        wait "${HEAD_PID}" || true
        exit 1
      fi
      sleep 2
    done

    touch '"${READY_FILE}"'
    wait "${HEAD_PID}"
  else
    DEADLINE=$((SECONDS + 240))
    while [ ! -f '"${READY_FILE}"' ]; do
      if [ "${SECONDS}" -ge "${DEADLINE}" ]; then
        echo "Timed out waiting for Ray head readiness file: '"${READY_FILE}"'"
        exit 1
      fi
      sleep 2
    done

    run_in_container
  fi
'
