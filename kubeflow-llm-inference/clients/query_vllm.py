# -*- coding: utf-8 -*-
"""
한국어 주석이 포함된 vLLM OpenAI 호환 엔드포인트 호출 예제 클라이언트입니다.
- 무엇을 하나요? KServe/Knative로 배포된 vLLM 서버의 /v1/chat/completions 엔드포인트를 호출합니다.
- Langfuse 연동? LANGFUSE_* 환경변수를 설정하면 요청/응답/지연시간을 Langfuse에 기록합니다.
- 자주 하는 실수: URL 뒤에 /v1/chat/completions 누락, Ingress 권한/Host 헤더 미설정.
"""
import os
import sys
import json
import time
import argparse
import requests
from langfuse_utils import get_langfuse, traced_observation


def main():
    # 인자 파서: 초보자용 설명을 덧붙였습니다.
    parser = argparse.ArgumentParser(description="KServe vLLM(OpenAI API) 호출 + Langfuse 트레이싱 예제")
    parser.add_argument("--url", required=True, help="모델 엔드포인트 URL (예: http://<ingress>/v1/chat/completions)")
    parser.add_argument("--prompt", default="Hello, Qwen!", help="모델에게 보낼 사용자 프롬프트")
    parser.add_argument("--model", default="qwen3-30b-fp8", help="vLLM에서 설정한 served-model-name")
    parser.add_argument("--max_tokens", type=int, default=256, help="생성할 최대 토큰 수")
    parser.add_argument("--temperature", type=float, default=0.2, help="샘플링 온도(0~1, 낮을수록 결정적)")
    parser.add_argument("--timeout", type=int, default=120, help="HTTP 요청 타임아웃(초)")
    args = parser.parse_args()

    # OpenAI ChatCompletions 규격에 맞춘 페이로드 구성
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

    # Langfuse 클라이언트 초기화(환경변수 없으면 None 반환)
    lf = get_langfuse()
    meta = {
        "model": args.model,
        "parameters": {"max_tokens": args.max_tokens, "temperature": args.temperature},
        "endpoint": args.url,
    }

    print(f"POST {args.url}")
    start = time.time()
    # Langfuse observation 컨텍스트로 요청/응답/지연시간을 기록
    with traced_observation(lf, name="client.chat.completions", metadata=meta) as obs:
        r = requests.post(args.url, headers=headers, data=json.dumps(payload), timeout=args.timeout)
        elapsed = time.time() - start
        # HTTP 오류를 예외로 승격(문제 원인 확인에 유용)
        r.raise_for_status()
        data = r.json()
        if obs is not None:
            usage = data.get("usage", {})  # vLLM이 반환하는 토큰 사용량(없을 수도 있음)
            obs.update(input=payload, output=data, metadata={"latency_s": elapsed, "usage": usage})
    # 보기 좋게 출력
    print(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()

