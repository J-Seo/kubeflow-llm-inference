# Megatron-LM v0.12 기반 분산 학습 컨테이너 (예시)
# 실제 운영 환경/쿠버네티스 노드의 CUDA/드라이버 버전에 맞춰 베이스 이미지를 조정하세요.
FROM nvcr.io/nvidia/pytorch:24.07-py3

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Megatron-LM v0.12 클론
RUN git clone --branch v0.12.0 https://github.com/NVIDIA/Megatron-LM.git megatron-lm

# Python 의존성
RUN pip install --no-cache-dir -U pip setuptools wheel
# 핵심 패키지: transformer-engine/apex는 환경에 따라 빌드가 필요합니다.
# 빌드 실패 시, 사전 빌드 이미지 사용 또는 호환 버전으로 조정하세요.
RUN pip install --no-cache-dir ninja packaging datasets transformers==4.* sentencepiece \
    langfuse==2.* tensorboard
# APEX (선택, 환경에 따라 빌드 필요)
RUN pip install --no-cache-dir --no-build-isolation git+https://github.com/NVIDIA/apex.git@master || true
# Transformer Engine (TE) - FP8/AMP 최적화. 환경에 맞게 버전 조정
RUN pip install --no-cache-dir transformer-engine==1.* || true

ENV PYTHONPATH=/workspace/megatron-lm:$PYTHONPATH

# 스크립트/설정 복사 (쿠버네티스에서 ConfigMap로 주입해도 무방)
COPY scripts/ /workspace/scripts/
COPY configs/ /workspace/configs/

# 기본 엔트리포인트는 PyTorchJob이 command/args로 지정하므로 생략

