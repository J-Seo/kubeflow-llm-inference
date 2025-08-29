# -*- coding: utf-8 -*-
"""
배치(여러 프롬프트) 추론을 수행하는 Kubeflow Pipelines 예제 (Langfuse 연동 포함)
- 컴포넌트 내에서 필요한 패키지를 설치하고, vLLM OpenAI 호환 엔드포인트를 호출합니다.
- Langfuse 자격이 주어지면 각 요청/배치 시작/종료를 관측 이벤트로 기록합니다.
주의: 운영 환경에서는 패키지를 이미지 빌드 시 포함시키는 것이 더 안정/빠릅니다.
"""
from typing import List
from kfp import dsl

@dsl.component(base_image="python:3.11-slim")
def batch_infer_component(endpoint: str, model: str, prompts: List[str], lf_host: str = "", lf_public: str = "", lf_secret: str = ""):
    import json, subprocess, time
    # 의존성 설치(간단 데모용). 운영에서는 미리 이미지를 빌드하세요.
    subprocess.check_call(["pip", "install", "--no-cache-dir", "requests==2.*", "langfuse==2.*"])  # langfuse Python SDK
    import requests
    from langfuse import Langfuse

    # Langfuse 초기화(자격 미제공 시 None)
    lf = None
    try:
        if lf_public and lf_secret:
            lf = Langfuse(public_key=lf_public, secret_key=lf_secret, host=(lf_host or "http://langfuse:3000"))
    except Exception:
        lf = None

    # 관측 헬퍼: 입력/출력/에러를 Langfuse observation으로 남깁니다.
    def log_obs(name, meta, inp=None, out=None, err=None):
        if lf is None:
            return
        o = lf.observation(name=name, metadata=meta)
        if inp is not None:
            o.update(input=inp)
        if out is not None:
            o.update(output=out)
        if err is not None:
            o.update(output=str(err), level="ERROR")
        o.end()

    batch_meta = {"endpoint": endpoint, "model": model, "num_prompts": len(prompts)}
    log_obs("pipeline.batch.start", batch_meta)

    # 각 프롬프트에 대해 OpenAI ChatCompletions 요청 수행
    for p in prompts:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": p},
            ],
            "max_tokens": 256,
            "temperature": 0.2,
        }
        t0 = time.time()
        try:
            r = requests.post(endpoint, json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            latency = time.time() - t0
            usage = data.get("usage", {})
            log_obs("pipeline.request", {"latency_s": latency, "usage": usage, "prompt": p}, inp=payload, out=data)
            print(json.dumps(data, indent=2, ensure_ascii=False))
        except Exception as e:
            log_obs("pipeline.request", {"error": str(e), "prompt": p}, inp=payload, err=e)
            raise

    log_obs("pipeline.batch.end", batch_meta)

@dsl.pipeline(name="vllm-batch-client")
def vllm_batch_client_pipeline(endpoint: str = "", model: str = "qwen3-30b-fp8", lf_host: str = "", lf_public: str = "", lf_secret: str = ""):
    # 데모용 프롬프트 2개
    prompts = [
        "Kubernetes에 대한 하이쿠를 써줘.",
        "Kubeflow를 쉽게 설명해줘.",
    ]
    batch_infer_component(endpoint, model, prompts, lf_host, lf_public, lf_secret)

