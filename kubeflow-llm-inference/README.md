# Kubeflow LLM Inference: Qwen3-30B FP8 on vLLM (KServe) and Ray

This repo contains production-ready manifests and examples to serve the Hugging Face model `Qwen/Qwen3-30B-A3B-Thinking-2507-FP8` on a 2-node GPU cluster (8x H100 per node, total 16 GPUs). Two deployment options are included:

- KServe + vLLM (recommended)
- vLLM on Ray (KubeRay)

It also includes a TGI example, a Kubeflow Pipelines batch client, and Langfuse observability.

## Hardware and layout
- 2 nodes with 8x H100 each (16 GPUs total)
- High-performance network (IB preferred) and RWX storage for model cache
- Kubernetes 1.27+, Kubeflow, KServe, and (optional) KubeRay installed

## Model size and memory
- Qwen3 30B FP8 weights are about ~60–70GB (FP8) + activation/kv cache. We provision generous memory and ephemeral storage. Adjust based on your environment.

## Structure
```
kubeflow-llm-inference/
├─ kserve/
│  ├─ vllm/
│  │  ├─ inference-service.yaml        # vLLM multi-GPU serving configuration
│  │  ├─ hf-secret.yaml                # Hugging Face token secret (optional)
│  │  └─ pvc-model.yaml                # Model cache PVC (optional)
│  └─ tgi/
│     └─ inference-service.yaml        # Text Generation Inference example
├─ pipelines/
│  └─ batch-client-pipeline.py         # Batch inference pipeline example
├─ clients/
│  ├─ query_vllm.py                    # Client to query serving endpoints
│  └─ langfuse_utils.py                # Shared Langfuse helpers for clients
├─ ray-option/                         # Multi-node distributed inference (vLLM+Ray)
│  ├─ raycluster.yaml                  # KubeRay cluster configuration
│  └─ vllm-ray-inference.yaml          # Ray backend vLLM serving
├─ observability/
│  └─ langfuse/
│     ├─ secrets.yaml                  # Langfuse + DB credentials and keys
│     ├─ postgres.yaml                 # Postgres for Langfuse with PVC
│     └─ langfuse.yaml                 # Langfuse server + service + ingress
```

## KServe + vLLM deployment
1. Create HF token secret and PVC (optional if you already have them):
   ```bash
   kubectl apply -f kubeflow-llm-inference/kserve/vllm/hf-secret.yaml
   kubectl apply -f kubeflow-llm-inference/kserve/vllm/pvc-model.yaml
   ```
2. (Optional) Deploy Langfuse observability stack:
   ```bash
   # Set secrets (edit secrets.yaml for domains/keys)
   kubectl apply -f kubeflow-llm-inference/observability/langfuse/secrets.yaml
   kubectl apply -f kubeflow-llm-inference/observability/langfuse/postgres.yaml
   kubectl apply -f kubeflow-llm-inference/observability/langfuse/langfuse.yaml
   ```
3. Deploy the InferenceService:
   ```bash
   kubectl apply -f kubeflow-llm-inference/kserve/vllm/inference-service.yaml
   ```
4. Wait for Ready:
   ```bash
   kubectl get isvc vllm-qwen3-30b-fp8 -w
   ```
5. Find the URL (Knative ingress):
   ```bash
   kubectl get isvc vllm-qwen3-30b-fp8 -o jsonpath='{.status.url}'
   ```
   The OpenAI endpoint will be `${URL}/v1/chat/completions`.

### vLLM + Langfuse integration
- vLLM pods include env vars for Langfuse (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST) and enable tracing (`VLLM_TRACING=1`).
- Client and pipeline are instrumented with the Langfuse SDK to capture request/response, usage, and latency.

## Client example
Use the included client to query the OpenAI endpoint:
```bash
export LANGFUSE_PUBLIC_KEY=... LANGFUSE_SECRET_KEY=... LANGFUSE_HOST=https://langfuse.example.com
python kubeflow-llm-inference/clients/query_vllm.py \
  --url "$(kubectl get isvc vllm-qwen3-30b-fp8 -o jsonpath='{.status.url}')/v1/chat/completions" \
  --model qwen3-30b-fp8 \
  --prompt "Explain what Kubeflow is in three sentences."
```

## TGI example (optional)
- `kubeflow-llm-inference/kserve/tgi/inference-service.yaml` demonstrates deploying TGI with 8 GPUs per replica (2 replicas = 16 GPUs). TGI may dequantize to FP16 at runtime.

## Ray option (vLLM + Ray)
1. Install KubeRay and create the cluster:
   ```bash
   kubectl apply -f kubeflow-llm-inference/ray-option/raycluster.yaml
   ```
2. Deploy the vLLM Ray head service (front-end):
   ```bash
   kubectl apply -f kubeflow-llm-inference/ray-option/vllm-ray-inference.yaml
   ```
3. Port-forward or expose the Service to access HTTP 8080 and metrics 8000.

## Monitoring, logging, and observability
- Prometheus annotations included for metrics scraping. If using kube-prometheus-stack, you can automatically discover metrics Services.
- Langfuse server is exposed via Ingress at `https://langfuse.example.com` (edit domain & TLS secret to match your environment).
- Access the Langfuse UI to view traces, breakdowns by latency and token usage, and drill down into individual requests.
- Logs: `kubectl logs -l app=vllm-qwen3-30b-fp8 -f`
- Health probes are configured for readiness/liveness/startup.

## Resource sizing and parallelism
- tensor-parallel-size=8 per replica; with 2 replicas you use 16 GPUs total. For pipeline parallelism, experiment with `--pipeline-parallel-size`.
- With Ray backend, `--tensor-parallel-size=16` creates a single logical 16-GPU instance across nodes.

## Model sharding
- vLLM handles sharding by tensor parallel within a pod. Ray option provides cross-node sharding into a unified instance.

## Storage
- PVC requests 3Ti RWX by default for model and cache artifacts. Adjust storageClassName and size.
- Postgres for Langfuse uses a 200Gi RWO PVC; tune to your retention needs.

## Security
- HF token is stored in a Kubernetes Secret (`hf-token`). Consider using a dedicated ServiceAccount and RBAC as needed.
- Langfuse credentials and DB DSN are stored in `langfuse-secrets` (change defaults!).

## Troubleshooting
- vLLM Pending: check nodeSelector labels, GPU quotas, PVC binding.
- OOM: reduce `--max-num-batched-tokens`, `--max-model-len`, or `--gpu-memory-utilization`.
- Langfuse 5xx: verify Postgres reachable and DATABASE_URL format; check NEXTAUTH_URL and domain/TLS.
- No traces: ensure LANGFUSE_* env vars set in client/pipeline and Langfuse URL reachable.


