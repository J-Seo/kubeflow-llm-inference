from typing import List
from kfp import dsl

# Batch client pipeline with Langfuse tracing for prompts list

@dsl.component(base_image="python:3.11-slim")
def batch_infer_component(endpoint: str, model: str, prompts: List[str], lf_host: str = "", lf_public: str = "", lf_secret: str = ""):
    import json, subprocess, time
    # Install dependencies
    subprocess.check_call(["pip", "install", "--no-cache-dir", "requests==2.*", "langfuse==2.*"])  # langfuse Python SDK
    import requests
    from langfuse import Langfuse

    lf = None
    try:
        if lf_public and lf_secret:
            lf = Langfuse(public_key=lf_public, secret_key=lf_secret, host=(lf_host or "http://langfuse:3000"))
    except Exception:
        lf = None

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
            print(json.dumps(data, indent=2))
        except Exception as e:
            log_obs("pipeline.request", {"error": str(e), "prompt": p}, inp=payload, err=e)
            raise

    log_obs("pipeline.batch.end", batch_meta)

@dsl.pipeline(name="vllm-batch-client")
def vllm_batch_client_pipeline(endpoint: str = "", model: str = "qwen3-30b-fp8", lf_host: str = "", lf_public: str = "", lf_secret: str = ""):
    prompts = [
        "Write a haiku about Kubernetes.",
        "Explain Kubeflow in simple terms.",
    ]
    batch_infer_component(endpoint, model, prompts, lf_host, lf_public, lf_secret)

