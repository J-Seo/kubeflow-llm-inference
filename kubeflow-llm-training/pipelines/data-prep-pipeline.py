from kfp import dsl

@dsl.component(base_image="python:3.11-slim")
def load_raw_dataset(hf_dataset: str, split: str) -> str:
    import subprocess, json
    subprocess.check_call(["pip", "install", "--no-cache-dir", "datasets==2.*"])  
    from datasets import load_dataset
    ds = load_dataset(hf_dataset, split=split)
    path = "/mnt/datasets/raw"
    ds.save_to_disk(path)
    return path

@dsl.component(base_image="python:3.11-slim")
def tokenize_dataset(raw_path: str, max_len: int = 4096) -> str:
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
        enc["labels"] = enc["input_ids"].copy()
        return enc
    proc = ds.map(map_fn, remove_columns=ds.column_names)
    out = "/mnt/datasets/processed"
    proc.save_to_disk(out)
    return out

@dsl.pipeline(name="llm-data-prep")
def data_prep_pipeline(hf_dataset: str, split: str = "train", max_len: int = 4096):
    p1 = load_raw_dataset(hf_dataset, split)
    p2 = tokenize_dataset(p1.output, max_len)

