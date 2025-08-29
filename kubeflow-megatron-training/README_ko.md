# Megatron-LM v0.12 기반 Qwen3-30B FP8 분산 학습 가이드 (한국어)

이 디렉터리는 NVIDIA Megatron-LM v0.12를 활용하여 Qwen/Qwen3-30B-A3B-Thinking-2507-FP8 모델을 2노드(각 8×H100, 총 16 GPU)에서 후처리/미세튜닝하는 예시를 제공합니다. 기존 `kubeflow-llm-training/`(Hugging Face Trainer/DeepSpeed)과 병행하여 사용할 수 있는 대안입니다.

## 구성 요소
```
kubeflow-megatron-training/
├─ Dockerfile                               # Megatron-LM v0.12 포함 컨테이너
├─ manifests/
│  └─ pytorchjob-megatron.yaml               # 2노드(16GPU) PyTorchJob 매니페스트
├─ scripts/
│  └─ train_qwen_megatron.py                 # Megatron 학습 런처(한국어 주석)
└─ configs/
   └─ megatron_qwen3_30b_2n16g.json          # 데이터/모델/병렬/런타임 설정(한국어 설명 포함)
```

## 사전 준비
- 쿠버네티스 + Kubeflow Training Operator(PytorchJob) 설치
- NVIDIA Device Plugin 설치, GPU 라벨(H100) 부여, NCCL IB/RDMA 권장
- RWX 스토리지: `llm-datasets-pvc`(데이터셋), `llm-checkpoints-pvc`(체크포인트) — 기존과 동일
- `hf-token` 시크릿 존재(HF 모델 접근)
- (선택) Langfuse 관측 서버: `langfuse-secrets` 시크릿

## 컨테이너 이미지 빌드
Dockerfile은 Megatron-LM v0.12를 클론하고 필요한 의존성을 설치합니다. 실환경에서는 APEX/Transformer Engine 빌드가 실패할 수 있으니, 사전 빌드 이미지 사용을 권장합니다.
```bash
# 레지스트리/태그는 환경에 맞게 변경
docker build -t <your-registry>/megatron-qwen:latest kubeflow-megatron-training/
docker push <your-registry>/megatron-qwen:latest
```

## 학습 설정 파일(configs/megatron_qwen3_30b_2n16g.json)
- data: 데이터셋 디렉터리(/mnt/datasets/processed), 시퀀스 길이, 배치 설정
- model: Qwen3-30B 구조/FP8/BF16 플래그, 어텐션/FFN 크기 등
- parallelism: TP=8, PP=2, DP는 PyTorch DDP 차원에서 형성(예시)
- runtime: 학습률, 웜업, 이터 수, 로깅/저장 주기, TensorBoard 경로

주의: Megatron은 전용 데이터 포맷을 사용합니다. `/mnt/datasets/processed`를 Megatron의 `--data-path` 형식(토큰화+인덱싱)으로 변환해야 합니다. 본 예시에서는 `.../megatron_text_document` 경로를 가정했습니다.

## PyTorchJob 배포
`manifests/pytorchjob-megatron.yaml`에서 이미지 이름을 빌드한 이미지로 바꾸고 배포합니다.
```bash
kubectl apply -f kubeflow-megatron-training/manifests/pytorchjob-megatron.yaml
kubectl get pytorchjob megatron-qwen3-30b-train -o yaml | less
kubectl logs -l role=master -f
kubectl logs -l role=worker -f
```

- torchrun: `--nnodes=2`, `--nproc_per_node=8`, rendezvous는 PyTorchJob의 서비스 DNS 사용
- PVC 마운트: `/mnt/datasets`, `/mnt/checkpoints` (기존과 동일)
- Langfuse 환경변수 설정 시, 학습 시작/종료/에러가 관측 이벤트로 기록됩니다.

## Megatron-LM 개념 요약
- Tensor Parallel(TP): 행렬 연산을 텐서 차원으로 분할. H100 8장 단위로 효과적
- Pipeline Parallel(PP): 레이어를 파이프라인 단계로 분리해 파이프라이닝
- Data Parallel(DP): 동일 모델 복제 후 데이터 배치 분산. 본 예시에서는 주로 TP/PP 사용
- Sequence Parallel: 시퀀스 차원 분할로 메모리/통신 최적화

## FP8 호환성
- Transformer Engine(TE)와 FP8 지원 조합 필요. 환경에 따라 TE/APEX 빌드가 필요하며, 실패 시 BF16-only로 운영하고 FP8 플래그를 제거하세요.

## 모니터링/관측
- TensorBoard: 체크포인트 PVC 하위의 `tb/` 경로 사용
- Langfuse: `LANGFUSE_PUBLIC_KEY/SECRET_KEY/HOST` 설정 시 학습 라이프사이클 이벤트 기록
- 추가로 Prometheus 노출/사이드카 구성을 도입할 수 있습니다(운영 정책에 맞게 확장).

## 문제 해결
- APEX/TE 빌드 실패: 사전 빌드 이미지 사용 권장, CUDA/드라이버/파이토치 버전 호환 확인
- NCCL 통신 오류: IB/RDMA 드라이버, NCCL_* 환경 변수, 파드 간 네트워크 경로 확인
- Megatron 데이터 포맷: `tools/preprocess_data.py` 등으로 토크나이저/인덱싱 처리 필요
- OOM: 글로벌 배치 축소, seq_length 감소, TP/PP 비율 조정, activation 체크포인트 사용 검토

## 검증/테스트 절차
- 짧은 `train_iters`와 작은 샘플 데이터로 드라이런
- `--log-interval`을 낮춰 초반 로깅 확인
- 체크포인트 저장/재개(resume) 동작 테스트

## 베스트 프랙티스
- 고정 버전 이미지로 재현성 확보
- 학습/데이터 변환 스크립트를 CI로 검증
- 스토리지/네트워크 성능 모니터링(노드/스토리지/네트워크 병목 파악)

## 참고 링크
- Megatron-LM: https://github.com/NVIDIA/Megatron-LM
- Transformer Engine: https://github.com/NVIDIA/TransformerEngine
- Kubeflow PyTorchJob: https://www.kubeflow.org/docs/components/training/operators/pytorch/

