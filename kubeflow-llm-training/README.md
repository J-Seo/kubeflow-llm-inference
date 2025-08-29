# Kubeflow LLM Training: Qwen3-30B FP8 Post-Training (QLoRA/Full FT)

This directory complements `kubeflow-llm-inference/` and provides a production-ready training setup on 2 nodes with 8x H100 GPUs per node (16 GPUs total).

## Structure
```
kubeflow-llm-training/
├─ manifests/
│  ├─ pvc-datasets.yaml                 # RWX PVCs for datasets & checkpoints
│  ├─ pytorchjob-qlora.yaml             # 2-node PyTorchJob for QLoRA fine-tuning
│  └─ configmap-train-scripts.yaml      # Training scripts + DeepSpeed configs
├─ pipelines/
│  └─ data-prep-pipeline.py             # KFP pipeline for dataset prep/tokenization
├─ observability/
│  └─ tensorboard.yaml                  # TensorBoard to visualize training logs
├─ katib/
│  └─ katib-experiment.yaml             # HPO example (placeholder)
└─ README.md
```

## Prereqs
- Kubeflow Training Operator (PyTorchJob CRD)
- GPUs: 2 nodes × 8x H100 each
- RWX storage class for PVCs
- Hugging Face token Secret (`hf-token`) present in namespace
- Langfuse installed (optional, for training telemetry)

## Data preparation
1. Create PVCs:
   ```bash
   kubectl apply -f kubeflow-llm-training/manifests/pvc-datasets.yaml
   ```
2. Compile and run the data prep pipeline (example):
   ```python
   from kfp import Client
   import kfp
   from kubeflow_llm_training.pipelines.data_prep_pipeline import data_prep_pipeline

   client = Client()
   client.create_run_from_pipeline_func(
       data_prep_pipeline,
       arguments={"hf_dataset": "Open-Orca/OpenOrca", "split": "train", "max_len": 4096},
   )
   ```
   Ensure the components mount the dataset PVC in your runtime environment if executed in-cluster.

## QLoRA training on 16 GPUs (2×8)
- Apply training scripts and PyTorchJob:
  ```bash
  kubectl apply -f kubeflow-llm-training/manifests/configmap-train-scripts.yaml
  kubectl apply -f kubeflow-llm-training/manifests/pytorchjob-qlora.yaml
  ```
- The job uses torchrun with `--nnodes=2` and `--nproc_per_node=8`.
- It installs required libs at startup (transformers, datasets, peft, deepspeed, accelerate, langfuse, bitsandbytes, tensorboard).
- Storage mounts:
  - /mnt/datasets: RWX dataset PVC
  - /mnt/checkpoints: RWX checkpoint PVC
- Scripts include:
  - `run_qlora.py` (LoRA/QLoRA)
  - `run_full_finetune.py` (full FT)
  - DeepSpeed configs: `ds_qlora.json`, `ds_full.json`

## TensorBoard
- Deploy and open TensorBoard:
  ```bash
  kubectl apply -f kubeflow-llm-training/observability/tensorboard.yaml
  kubectl port-forward svc/tensorboard-llm 6006:80
  ```
  Then visit http://localhost:6006.

## Langfuse training telemetry
- Training scripts try to initialize Langfuse from env vars (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST) and log observations on logging/save.
- Ensure the `langfuse-secrets` Secret exists (see inference README) or export env vars if running locally.

## Resource management and GPU optimization
- Requests/limits set for 8 GPUs per pod (Master/Worker) and high memory.
- NCCL settings tuned for IB/RDMA; adjust for your fabric.
- Checkpoints saved every `--save_steps` and limited via `--save_total_limit`.
- For FP8-aware models, training often runs with bf16 compute; adjust ds configs and args if needed.

## Best practices
- Start with QLoRA on a small dataset to validate throughput/stability, then scale.
- Use gradient_accumulation to fit larger seq lengths; monitor GPU mem.
- Keep datasets/tokenization consistent with inference tokenizer.
- Validate outputs by deploying the new adapter with the existing inference stack.

## Troubleshooting
- Pending pods: check node labels, GPU availability, and PVC binding.
- OOM: reduce batch size, GA, seq length, or zero stage.
- NCCL: verify IB drivers; consider setting NCCL_TOPO_FILE for topology-aware tuning.
- Slow startup: image pulls & pip installs; consider baking a custom image with deps and scripts preinstalled.

## Integration with inference
- After training, merge adapters or reference the checkpoint path from the inference manifests (e.g., swap model to `/mnt/checkpoints/qwen3-30b-qlora` or push to a registry/Hub).
- Keep model name consistent (`served-model-name`) when updating inference configs.

