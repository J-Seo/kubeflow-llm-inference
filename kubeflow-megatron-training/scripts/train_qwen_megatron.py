# -*- coding: utf-8 -*-
"""
Megatron-LM v0.12 + Qwen3-30B FP8 학습 실행 스크립트(한국어 주석)
- 목적: Megatron의 분산 병렬(TP/PP/DP)을 활용해 2노드 16GPU로 학습/후처리
- 입력: configs/megatron_qwen3_30b_2n16g.json (데이터/모델/병렬/런타임 설정)
- 출력: /mnt/checkpoints/* (체크포인트 및 TensorBoard 로그)
주의: Megatron-LM의 공식 API는 버전별로 바뀔 수 있습니다. v0.12 기준 예시이며, 환경에 맞춰 수정하세요.
"""
import os
import json
import argparse
import subprocess

# (선택) Langfuse 관측: 학습 시작/로그/저장을 관측으로 남길 수 있습니다.
try:
    from langfuse import Langfuse
except Exception:
    Langfuse = None


def get_langfuse():
    pub = os.getenv("LANGFUSE_PUBLIC_KEY")
    sec = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST")
    if not (pub and sec and host) or Langfuse is None:
        return None
    try:
        return Langfuse(public_key=pub, secret_key=sec, host=host)
    except Exception:
        return None


def parse_args():
    p = argparse.ArgumentParser(description="Megatron-LM v0.12 Qwen3-30B FP8 Training Launcher")
    p.add_argument("--config", default="/workspace/configs/megatron_qwen3_30b_2n16g.json", help="설정 JSON 경로")
    p.add_argument("--output_dir", default="/mnt/checkpoints/qwen3-30b-megatron", help="체크포인트 출력 경로")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config, "r") as f:
        cfg = json.load(f)

    # Megatron 파라미터 구성 (예시)
    data = cfg["data"]; model = cfg["model"]; parr = cfg["parallelism"]; rt = cfg["runtime"]

    # Megatron pretrain_gpt.py 호출 인자 구성
    script = "/workspace/megatron-lm/pretrain_gpt.py"

    base_args = [
        "python", script,
        "--tensor-model-parallel-size", str(parr["tensor_model_parallel_size"]),
        "--pipeline-model-parallel-size", str(parr["pipeline_model_parallel_size"]),
        "--sequence-parallel" if parr.get("sequence_parallel", False) else "",
        "--num-layers", str(model["num_layers"]),
        "--hidden-size", str(model["hidden_size"]),
        "--num-attention-heads", str(model["num_attention_heads"]),
        "--seq-length", str(data["seq_length"]),
        "--max-position-embeddings", str(data["seq_length"]),
        "--micro-batch-size", str(data["micro_batch_size"]),
        "--global-batch-size", str(data["global_batch_size"]),
        "--lr", str(rt["lr"]),
        "--min-lr", str(rt["min_lr"]),
        "--weight-decay", str(rt["weight_decay"]),
        "--adam-beta1", str(rt["adam_beta1"]),
        "--adam-beta2", str(rt["adam_beta2"]),
        "--adam-eps", str(rt["adam_eps"]),
        "--lr-warmup-steps", str(rt["lr_warmup_steps"]),
        "--train-iters", str(rt["train_iters"]),
        "--log-interval", str(rt["log_interval"]),
        "--save-interval", str(rt["save_interval"]),
        "--save", args.output_dir,
        "--tensorboard-dir", rt.get("tensorboard_dir", os.path.join(args.output_dir, "tb")),
        "--bf16" if model.get("bf16", False) else "",
        # FP8은 환경 준비 완료 후에만 켜세요(TE/APEX 호환성 필요).
        # "--fp8" if model.get("fp8", False) else "",
        "--data-path", os.path.join(data["dataset_dir"], "megatron_text_document")  # Megatron 포맷 가정
    ]
    # 빈 문자열 제거
    base_args = [a for a in base_args if a]

    # Qwen3-30B 모델 로딩 관련 옵션 (예: HF 체크포인트에서 로드)
    # Megatron v0.12에서 직접 HF를 로드하는 예제는 제한적이므로, 사전 변환(가중치 변환)을 추천.
    # 여기서는 예시로 모델 이름을 메타데이터로 전달합니다.
    base_args += ["--init-from-hf", model["model_name_or_path"], "--use-checkpoint-args"]

    # 환경 변수 (NCCL/네트워크)
    env = os.environ.copy()
    env.setdefault("NCCL_DEBUG", "WARN")
    env.setdefault("NCCL_IB_DISABLE", "0")
    env.setdefault("NCCL_P2P_DISABLE", "0")

    lf = get_langfuse()
    if lf:
        obs = lf.observation(name="megatron.train.start", metadata={"config": cfg})
        obs.end()

    # Megatron 스크립트 실행
    print("Launching Megatron-LM training with args:\n", " ".join(base_args))
    try:
        subprocess.check_call(base_args, env=env)
    except subprocess.CalledProcessError as e:
        print("[ERROR] Megatron training failed:", e)
        if lf:
            o = lf.observation(name="megatron.train.error", metadata={"returncode": e.returncode})
            o.update(output=str(e), level="ERROR"); o.end()
        raise
    else:
        if lf:
            o = lf.observation(name="megatron.train.end", metadata={"output_dir": args.output_dir})
            o.end()


if __name__ == "__main__":
    main()

