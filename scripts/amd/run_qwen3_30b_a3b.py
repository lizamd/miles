from dataclasses import dataclass
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_minimal"] = "normal"
    run_id: str = U.create_run_id()
    model_name: str = "Qwen3-30B-A3B"
    megatron_model_type: str = "qwen3-30B-A3B"
    num_gpus_per_node: int | None = None
    hardware: Literal["auto", "MI350X", "MI355X", "MI455X"] = "auto"
    enable_eval: bool = True
    extra_args: str = ""
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"
    rollout_fp8: bool = False
    train_fp8: bool = False
    enable_megatron_bridge: bool = False
    enable_mis: bool = False
    # TODO improve, should be able to override more easily
    tis_use_rs: bool = True

    def __post_init__(self):
        self.hardware = U.resolve_hardware(self)
        self.num_gpus_per_node = self.num_gpus_per_node or U.NUM_GPUS_OF_HARDWARE[self.hardware]


def prepare(args: ScriptArgs):
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    U.exec_command_cpu(f"hf download Qwen/{args.model_name} --local-dir {args.model_dir}/{args.model_name}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
    U.hf_download_dataset("zhuzilin/aime-2024", data_dir=args.data_dir)

    if args.rollout_fp8:
        U.exec_command_cpu(
            f"hf download Qwen/{args.model_name}-FP8 --local-dir {args.model_dir}/{args.model_name}-FP8"
        )

    if not args.enable_megatron_bridge:
        U.convert_checkpoint(
            model_name=args.model_name,
            megatron_model_type=args.megatron_model_type,
            num_gpus_per_node=args.num_gpus_per_node,
            # To support multi-node training, for simplicity, we put model into shared folder
            dir_dst=args.model_dir,
            hf_checkpoint=f"{args.model_dir}/{args.model_name}",
            megatron_path=args.megatron_path,
        )


# TODO improve layering: split algorithm vs infra
def execute(args: ScriptArgs):
    ref_load_path = (
        f"{args.model_dir}/{args.model_name}/"
        if args.enable_megatron_bridge
        else f"{args.model_dir}/{args.model_name}_torch_dist"
    )
    load_save_path = f"{args.output_dir}/{args.run_id}/checkpoints"

    if args.rollout_fp8:
        hf_checkpoint = f"{args.model_dir}/{args.model_name}-FP8"
    else:
        hf_checkpoint = f"{args.model_dir}/{args.model_name}"
    ckpt_args = (
        f"--hf-checkpoint {hf_checkpoint}/ "
        f"--ref-load {ref_load_path} "
        f"--load {load_save_path} "
        f"--save {load_save_path} "
        f"--save-interval {2 if args.mode == 'debug_minimal' else 20} "
        f"--save-retain-interval {2 if args.mode == 'debug_minimal' else 20} "
    )

    rollout_args = (
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 3000 "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 8 "
        f"--rollout-max-response-len {100 if args.mode == 'debug_minimal' else 8192} "
        "--rollout-temperature 1 "
        "--global-batch-size 256 "
        "--balance-data "
    )

    eval_args = ""
    if (args.mode != "debug_minimal") and args.enable_eval:
        eval_args += (
            "--eval-interval 20 "
            f"--eval-prompt-data aime {args.data_dir}/aime-2024/aime-2024.jsonl "
            "--n-samples-per-eval-prompt 16 "
            "--eval-max-response-len 16384 "
            "--eval-top-p 1 "
        )

    perf_args = (
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        # "--micro-batch-size 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 32768 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    # gfx1250 has no flash-attn build, and Megatron's --attention-backend flash sets
    # NVTE_FLASH_ATTN=1 / NVTE_FUSED_ATTN=0 / NVTE_UNFUSED_ATTN=0, pinning Transformer Engine
    # to a backend that is not installed; training then dies with "No dot product attention
    # backend is available for the provided inputs". auto enables all three and lets TE pick
    # what exists, which on gfx1250 is UnfusedDotProductAttention -- it handles the packed/THD
    # layout fine. TE's fused attention is unavailable on this part both here and in AMD's own
    # Primus gfx1250 image, so auto resolving to unfused is the expected outcome, not a
    # degradation introduced by this variant.
    attention_backend = "auto" if args.hardware == "MI455X" else "flash"

    # gfx1250 rollout backend. sglang's own mi45x CI pairs a triton prefill with an aiter
    # decode (SGLANG_USE_AITER_UNIFIED_ATTN=1), and that is the faster combination, but the
    # aiter kernel reads out of bounds on this part: two runs died in
    #   kernel_unified_attention_3d_num_query_heads_16_num_queries_per_kv_4_BLOCK_SIZE_64_
    #   TILE_SIZE_64_HEAD_SIZE_128_NUM_SEGMENTS_PER_SEQ_4_..._ALL_DECODE_1_...
    # with "Memory access fault ... Page not present", at 65 concurrent requests and 6 % KV
    # cache use, which takes both sglang schedulers down with exit -6 and fails the job.
    # Neither concurrency nor memory pressure is the trigger. triton for both phases is
    # slower and is what actually completes a run; revisit once the kernel is fixed upstream.
    rollout_attention_args = (
        "--sglang-attention-backend triton " if args.hardware == "MI455X" else ""
    )

    misc_args = (
        # default dropout in megatron is 0.1
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        # should be good for model performance
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        # need to comment this when using model with MLA
        f"--attention-backend {attention_backend} "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--colocate "
        "--use-fault-tolerance "
        f"--dump-details {args.output_dir}/{args.run_id}/dump_details "
    )
    misc_env_vars = {}

    if args.train_fp8:
        misc_args += (
            "--transformer-impl transformer_engine "
            "--bf16 "
            "--fp8-format e4m3 "
            "--fp8-recipe blockwise "
            "--no-gradient-accumulation-fusion "
        )
        misc_env_vars |= {
            "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": "0",
            "GPU_MAX_HW_QUEUES": "1",
            # keep Ray from blanking HIP/CUDA visibility for the job entrypoint
            "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES": "1",
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
        }

    if args.enable_megatron_bridge:
        misc_args += "--megatron-to-hf-mode bridge "

    match (args.hardware, args.num_nodes):
        # A 4-GPU MI455X node cannot use the MI355X shape. Megatron requires
        # expert_tensor_parallel x expert_model_parallel x pipeline_model_parallel to divide
        # world size, and the MI355X profile's 1 x 4 x 2 = 8 does not divide 4:
        #   RuntimeError: world_size (4) is not divisible by
        #                 expert_tensor_model_pipeline_parallel size (8)
        # PP=2 with EP=2 makes that product 4. Keeping PP=2 matters for host memory, not just
        # divisibility: each rank then holds half the 48 layers, and the first attempt here
        # (PP=1, EP=4) needed ~44 GB of host RAM per rank to build and load the model -- 176 GB
        # across four ranks on a 251 GB box, which Ray's OOM killer cut down. The per-rank
        # expert count is unchanged by the trade: (128/EP) x (48/PP) is 64 x 24 either way,
        # the same 1536 expert-layers the 8-GPU MI355X profile gives each of its ranks.
        # TP stays 1, so no --sequence-parallel (it needs TP > 1). DP is 2.
        # Enablement only: nothing here has been tuned for 432 GB cards.
        case ("MI455X", 1):
            perf_args += (
                "--tensor-model-parallel-size 1 "
                "--pipeline-model-parallel-size 2 "
                "--context-parallel-size 1 "
                "--expert-model-parallel-size 2 "
                "--expert-tensor-parallel-size 1 "
                "--max-tokens-per-gpu 16384 "
            )
            sglang_args = (
                "--rollout-num-gpus-per-engine 2 "
                "--sglang-mem-fraction-static 0.7 "
                "--sglang-max-running-requests 512 "
            )
            # No --optimizer-cpu-offload here, unlike MI355X. That trade only makes sense when
            # HBM is the scarce side: a 288 GB MI355X node has 8 cards against the same host
            # RAM, while gfx1250 has 432 GB per card and this node has 4 of them. Offloading
            # put ~45 GB of optimizer state per rank into host memory -- ~170 GB across four
            # ranks on a 251 GB machine -- and Ray's OOM killer took out two of the actors.
            optimizer_args += "--use-precision-aware-optimizer "
        case ("MI350X" | "MI355X", 1 | 2):
            perf_args += (
                "--tensor-model-parallel-size 1 "
                "--sequence-parallel "
                "--pipeline-model-parallel-size 2 "
                "--context-parallel-size 2 "
                "--expert-model-parallel-size 4 "
                "--expert-tensor-parallel-size 1 "
                "--max-tokens-per-gpu 16384 "
            )
            sglang_args = (
                "--rollout-num-gpus-per-engine 2 "
                "--sglang-mem-fraction-static 0.7 "
                "--sglang-max-running-requests 512 "
            )
            optimizer_args += (
                "--optimizer-cpu-offload " "--overlap-cpu-optimizer-d2h-h2d " "--use-precision-aware-optimizer "
            )
        case _:
            raise NotImplementedError

    if args.enable_mis:
        config_text = f"""
use_tis: true
use_rs: {"true" if args.tis_use_rs else "false"}
tis_level: "token"
rs_level: "token"
tis_mode: "truncate"
tis_lower_bound: 0.5
tis_upper_bound: 2.0
rs_lower_bound: null
rs_upper_bound: null
rs_veto_threshold: 1.0e-4
tis_batch_normalize: true
""".strip()
        misc_args += (
            f"--custom-config-path {U.encode_pseudo_file(config_text)} "
            "--custom-tis-function-path examples.infra_features.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp "
        )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "

        f"{rollout_attention_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        extra_env_vars={**misc_env_vars},
        megatron_path=args.megatron_path,
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
