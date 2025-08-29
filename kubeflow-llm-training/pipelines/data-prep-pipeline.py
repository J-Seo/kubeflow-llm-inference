# -*- coding: utf-8 -*-
"""
데이터 준비 파이프라인(Kubeflow Pipelines)
- load_raw_dataset: 허깅페이스 데이터셋을 다운로드하고 디스크에 저장
- tokenize_dataset: 모델 토크나이저로 토큰화하여 학습 형식으로 저장(labels 포함)
주의: PVC 마운트 경로(/mnt/datasets)가 실제 런타임 환경과 일치해야 합니다.
"""
from kfp import dsl

@dsl.component(base_image="python:3.11-slim")
def load_raw_dataset(hf_dataset: str, split: str) -> str:
    """HF 데이터셋을 지정된 split으로 로드하고 /mnt/datasets/raw 경로에 저장합니다.
    - hf_dataset 예: "Open-Orca/OpenOrca"
    - split 예: "train", "validation"
    """
    import subprocess, json
    subprocess.check_call(["pip", "install", "--no-cache-dir", "datasets==2.*"])
    from datasets import load_dataset
    ds = load_dataset(hf_dataset, split=split)
    path = "/mnt/datasets/raw"
    ds.save_to_disk(path)
    return path

@dsl.component(base_image="python:3.11-slim")
def tokenize_dataset(raw_path: str, max_len: int = 4096) -> str:
    """저장된 raw 데이터셋을 불러와 모델 토크나이저로 토큰화하여 저장합니다.
    - max_len: 시퀀스 최대 길이(메모리/성능과 직결)
    """
    import subprocess
    subprocess.check_call(["pip", "install", "--no-cache-dir", "datasets==2.*", "transformers==4.*"])
    from datasets import load_from_disk
    from transformers import AutoTokenizer
    import os
    model = os.getenv("MODEL_NAME", "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8")
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ds = load_from_disk(raw_path)
    def map_fn(ex):
        text = ex.get("text") or ex.get("input")
        enc = tok(text, max_length=max_len, truncation=True, padding="max_length")
        enc["labels"] = enc["input_ids"].copy()  # Causal LM 학습용 레이블 생성
        return enc
    proc = ds.map(map_fn, remove_columns=ds.column_names)
    out = "/mnt/datasets/processed"
    proc.save_to_disk(out)
    return out

@dsl.pipeline(name="llm-data-prep")
def data_prep_pipeline(hf_dataset: str, split: str = "train", max_len: int = 4096):
    """데이터 준비 파이프라인: 원천 로드 → 토큰화.
    제출 시 arguments로 hf_dataset/split/max_len을 전달하세요.
    """
    p1 = load_raw_dataset(hf_dataset, split)
    p2 = tokenize_dataset(p1.output, max_len)

