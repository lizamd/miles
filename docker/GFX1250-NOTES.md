# miles on gfx1250 / MI455X — state, findings, and what is left

Written before a node reboot on 2026-09-05. Everything below was measured on
`ctheliosp-1b112-a37-1` (4x gfx1250, 432 GB/card, 251 GB host RAM).

## What exists

* Image `rocm/sgl-dev:mi45x-dev-rocm10-mi45x`, built from the branch below. Docker's
  store is on `/dev/nvme0n1p2`, so it survives the reboot.
* Branch `feat/rocm10-mi45x` in `/home/lizli102/work/miles`, commit `fe1846b`, not
  pushed. It adds a `rocm10-mi45x` variant to `docker/build.py` +
  `docker/Dockerfile.rocm` and an MI455X profile to the Qwen3 scripts. The commit
  message explains each fix; the scripts under `docker/` carry the reasoning in full.
* Prebuilt wheels in `docker/prebuilt/`: Transformer Engine 2.17.0+ebbd623b2 (62 MB)
  and apex 1.15.0a0 (84 MB), both gfx1250 + ROCm 10, rebuildable with
  `docker/build_te_gfx1250.sh` and `docker/build_apex_gfx1250.sh` **on a machine with
  a GPU** — apex's setup.py imports torch, which pulls in aiter, which runs rocminfo.
* Validation scripts in `/home/lizli102/work/validate/`: `v1_single.py`,
  `v2_rccl.py` (1 KB-256 MB all_reduce + all_gather), `v3_megatron.py` (4-GPU TP=2/DP=2
  GPTModel, 5 steps).
* Models and datasets under `/home/lizli102/work/models` and `/datasets`:
  Qwen3-30B-A3B (57 GB) + its torch_dist conversion (27 GB), Qwen3-4B (7.6 GB) + its
  conversion (7.5 GB), dapo-math-17k, aime-2024.

## Verified on hardware

All five passed while the GPUs were healthy:

| check | result |
|---|---|
| single GPU: arch, TE bf16 + fp8 fwd/bwd, aiter, sglang, megatron, miles imports | pass |
| `torch.cuda.device_count()` and miles `detect_hardware()` -> MI455X, 4 GPUs | pass |
| 4-GPU RCCL, 1 KB to 256 MB all_reduce + all_gather, values checked | pass |
| 4-GPU Megatron GPTModel, 5 steps, loss 8.5030 -> 8.4222, all finite | pass |
| apex `wgrad_gemm_accum_fp32` vs a torch reference | max abs err 5.7e-06 |

## Run-time settings gfx1250 needs

The first five are baked into the branch. The last is a `docker run` flag and is not.

| setting | value | why |
|---|---|---|
| `--attention-backend` | `auto` | `flash` sets NVTE_FLASH_ATTN=1 / FUSED=0 / UNFUSED=0, pinning TE to a backend with no gfx1250 build |
| `--sglang-prefill-attention-backend` | `triton` | sglang's own mi45x CI |
| `--sglang-decode-attention-backend` | `aiter` | same; the unified kernel, not the old paged path |
| `SGLANG_USE_AITER_UNIFIED_ATTN` | `1` | without it decode lands in `paged_attention_ragged`, whose arch list (`csrc/cpp_itfs/utils.py`, upstream too) has no gfx1250 |
| `ENABLE_CK` | `0` | no CK kernels for gfx1250 |
| `docker run --ulimit nofile` | >= 65536 | the default 1024 kills raylet with "Too many open files" during Ray startup |

## Qwen3-4B runs end to end

`run_qwen3_4b.py --num-rollout 2 --no-enable-eval` completed on 4x gfx1250 after the
reboot, with zero faults. The log shows the full loop, not just generation:

```
rollout_id=0 -> num_rollouts=[64] -> rollout_id=1
Weight version changed. old_version='2' new_version='3'
Memory-Usage before/after update_weights
Job 'raysubmit_zULYnQFU6qWVEhG1' succeeded
```

The weight version advancing is what proves a training step ran and produced new
weights that were pushed back into the inference engines.

It needed `--sglang-attention-backend triton`. See below.

## Qwen3-30B-A3B runs end to end too

Async, not colocate. `run_qwen3_30b_a3b.py --no-enable-eval` with a reduced rollout
(`--rollout-batch-size 8 --global-batch-size 64 --rollout-max-response-len 2048`) completed
two rollouts and four training steps on 4x gfx1250, weight broadcasts at 4-6 s, no deadlocks,
faults or timeouts.

Training and inference agree numerically, which matters here because every kernel differs
between the two sides -- TE Unfused attention and Megatron MoE against sglang's triton
attention and triton MoE:

| step | train_rollout_logprob_abs_diff | train_rollout_kl |
|---|---|---|
| 0 | 0.0232 | 0.00237 |
| 1 | 0.0198 | 0.00212 |
| 2 | 0.0176 | 0.00178 |

Both fall monotonically. 0.02 absolute logprob difference is what bf16 inference costs; the
point is that it is small and stable rather than drifting, which is what says the triton
fallbacks are trustworthy and not merely non-crashing. A drift here would break the policy
gradient silently, without any error.

## Two things gfx1250 needs that took the longest to find

**Async, not colocate, on a 4-GPU node.** Colocate makes `--offload-train` default to true,
which parks the whole model in host RAM whenever the rollout engines want the GPUs: each of
four ranks went from 24.9 GB to 44.0 GB of RSS, 176 GB against 251 GB of system memory, and
Ray's OOM killer took the actors out. HBM was never the constraint -- 432 GB per card, 170 GB
still free. Splitting the GPUs two and two removes the offload entirely; host use dropped to
53 GB.

**Primus's collective settings.** Without them the first weight broadcast hangs for the full
1800 s watchdog timeout and the job dies. They are baked into the variant now; see the
Dockerfile. `NCCL_IB_DISABLE=1` is *not* baked in -- it is a single-node choice -- so pass it
at run time on one node.

## Open: an out-of-bounds read in aiter's unified attention kernel

sglang's own mi45x CI pairs a triton prefill with an aiter decode
(`SGLANG_USE_AITER_UNIFIED_ATTN=1`). That configuration reads out of bounds on gfx1250.
Two faults in one run, on two different GPUs, both here:

```
kernel_unified_attention_3d_num_query_heads_16_num_queries_per_kv_4_BLOCK_SIZE_64_
TILE_SIZE_64_HEAD_SIZE_128_NUM_SEGMENTS_PER_SEQ_4_num_warps_1_waves_per_eu_1_
num_stages_2_ALL_DECODE_1_SHUFFLED_KV_CACHE_0_IS_Q_FP8_0_IS_KV_FP8_0

Memory access fault ... Reason: Page not present or supervisor privilege
```

Both sglang schedulers then died with exit -6 and the job failed. Reproduced with
Qwen3-4B, TP=2, ROCm 10.0.0, aiter a6d2b564 — enough detail to file upstream.

Neither concurrency nor memory is the trigger, contrary to what the first fault
suggested: it hit at 65 concurrent requests and 6 % KV cache use, against 201 and 1 %
the first time. An earlier note here blamed the GEMM whose shape line happened to print
alongside; the kernel name above is what the fault actually reports.

The scripts now pass `--sglang-attention-backend triton` for MI455X, which is slower and
is what completes a run. Revisit when the kernel is fixed.

Concurrency, if you need it pinned: it is emergent, bounded by
`sglang_server_concurrency * rollout_num_gpus // rollout_num_gpus_per_engine`
(`miles/rollout/inference_rollout/inference_rollout_common.py:40`) on the client and by
`max_running_requests` on the server, which defaults to at least 2048
(`mem_cache/kv_cache_configurator.py:1873`). Pinning one side alone does nothing.

## Why the node is being rebooted

The fault wedged the device. dmesg shows 58 kernel-context faults (`vmid:0 pasid:0`,
the command-processor signature in GFX1250_RUNBOOK-v2 §5.3) starting at 03:57:05,
right after the 03:54 fault. From then on every multi-process run hung in
`torch.distributed.new_group`, including a plain 4-GPU RCCL test that had passed
repeatedly before and that has nothing to do with miles. Single-process GEMMs still
worked, which is what made this easy to miss.

`rocm-smi --gpureset` reset GPUs 0 and 1 and then hung on GPU 2 for 19+ minutes.

**A reboot restores the device; it does not fix the kernel defect.** The same path
will fault again. Cap concurrency on both sides before the next rollout run, and if it
faults anyway, record the exact `[aiter] shape is M:..,N:..,K:..` line — that is the
reproducible detail worth reporting upstream.

## Left undone

1. **A long 30B run.** Two rollouts is not training. The rollout was cut to 64 samples at
   2048 tokens to reach the broadcast quickly; at the script's own 256 samples at 8192 the
   first rollout took 36 minutes and the training step 64, which does not scale to
   `--num-rollout 3000` on four GPUs. Somebody needs to decide what size run is worth doing
   here.
2. **Push.** Nothing has been pushed and the image has not been published.
   `python docker/build.py --variant rocm10-mi45x --image-tag dev --push` would put it
   at `rocm/sgl-dev:miles-rocm10-mi45x`.
3. **The two wheels in git.** 145 MB of binaries sit in `docker/prebuilt/`. Upstream
   they belong in a `miles-wheels-rocm` release, reached via `WHEELS_TAG_ROCM`, the way
   the gfx950 variants get theirs.

## Post-reboot checklist

```bash
sudo modprobe amdgpu noretry=0 gpu_recovery=0 ip_block_mask=0xcff
sudo cat /sys/class/drm/card*/device/ualink/accel_state
export ROCM_PATH=/opt/rocm PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib/rocm_sysdeps/lib:$LD_LIBRARY_PATH
export HIP_DEVICE_LIB_PATH=$ROCM_PATH/lib/llvm/amdgcn/bitcode

# device is healthy only if this passes; it is what caught the wedge
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
  --security-opt seccomp=unconfined --ipc=host --shm-size=32g \
  --ulimit nofile=1048576:1048576 -v /home/lizli102/work/validate:/v \
  --entrypoint bash -w /v rocm/sgl-dev:mi45x-dev-rocm10-mi45x \
  -c 'torchrun --nproc_per_node=4 v2_rccl.py'
```
