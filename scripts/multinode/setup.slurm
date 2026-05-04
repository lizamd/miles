#!/bin/bash
#SBATCH --job-name=miles-multinode-setup
#SBATCH --nodelist=mia1-p01-g[05-06]
#SBATCH --nodes=2
#SBATCH --partition=amd-aim
#SBATCH --account=amd-aim
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/setup_%j.log
#SBATCH --error=logs/setup_%j.err

set -euo pipefail

# Path to the host directory that contains the miles repo (becomes /workspace inside container).
# Override via:  WORKSPACE=/path/to/host-dir sbatch setup.slurm
WORKSPACE="${WORKSPACE:-/home/liz.li@amd.com}"
IMAGE="${IMAGE:-rlsys/miles:MI350-355-latest}"
CONTAINER="${CONTAINER:-miles-dev}"

mkdir -p "${WORKSPACE}/logs"

# --------------------------------------------------------------------------- #
#  Per-node setup: launch container and run the init script inside it         #
# --------------------------------------------------------------------------- #
srun bash -c '
  set -euo pipefail
  echo "[$(hostname)] Starting container setup ..."

  docker rm -f '"${CONTAINER}"' 2>/dev/null || true

  # Start container detached so it stays running after setup
  docker run -d \
    --network=host \
    --ipc=host \
    --device /dev/dri \
    --device /dev/kfd \
    --device /dev/infiniband \
    --group-add video \
    --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --privileged \
    -v $HOME/.ssh:/root/.ssh \
    -v '"${WORKSPACE}"':/workspace \
    --shm-size 128G \
    --name '"${CONTAINER}"' \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --ulimit nofile=1048576:1048576 \
    -w /workspace \
    '"${IMAGE}"' \
    sleep infinity

  # Run setup inside the running container
  docker exec '"${CONTAINER}"' bash /workspace/miles/scripts/multinode/setup_node.sh

  echo "[$(hostname)] Done."
'
