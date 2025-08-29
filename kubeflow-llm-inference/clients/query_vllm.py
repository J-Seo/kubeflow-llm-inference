import os
import sys
import json
import time
import argparse
import requests
from langfuse_utils import get_langfuse, traced_observation


def main():
    parser = argparse.ArgumentParser(description="Query vLLM OpenAI server via KServe with Langfuse tracing")
    parser.add_argument("--url", required=True, help="Model endpoint URL, e.g., http://<ingress>/v1/chat/completions")
    parser.add_argument("--prompt", default="Hello, Qwen!", help="Prompt to send")
    parser.add_argument("--model", default="qwen3-30b-fp8", help="served-model-name for vLLM")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    headers = {"Content-Type": "application/json"}
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": args.prompt},
        ],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": False,
    }

    lf = get_langfuse()
    meta = {
        "model": args.model,
        "parameters": {"max_tokens": args.max_tokens, "temperature": args.temperature},
        "endpoint": args.url,
    }

    print(f"POST {args.url}")
    start = time.time()
    with traced_observation(lf, name="client.chat.completions", metadata=meta) as obs:
        r = requests.post(args.url, headers=headers, data=json.dumps(payload), timeout=args.timeout)
        elapsed = time.time() - start
        r.raise_for_status()
        data = r.json()
        if obs is not None:
            usage = data.get("usage", {})
            obs.update(input=payload, output=data, metadata={"latency_s": elapsed, "usage": usage})
    print(json.dumps(data, indent=2))
    print(f"Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()

