# Kubeflow 기반 대규모 언어모델 학습/후처리 가이드 (한국어)

본 문서는 `kubeflow-llm-training/` 디렉터리의 구성을 한국어로 상세히 설명합니다. Qwen/Qwen3-30B-A3B-Thinking-2507-FP8 모델을 2개 노드(각 8×H100)에서 QLoRA 또는 전체 파인튜닝으로 후처리(포스트 트레이닝)하는 방법을 안내합니다.

## 1. 요구 사항
- Kubeflow Training Operator(PytorchJob CRD) 설치
- GPU 노드 2개(각 H100 8장), 충분한 CPU/메모리
- RWX 스토리지 클래스(NFS/CSI 등) — 데이터셋/체크포인트 공유
- Hugging Face 토큰 시크릿(`hf-token`) 존재
- (선택) Langfuse 관측 서버 배포(인퍼런스 README 참조)

## 2. 디렉터리 개요
```
kubeflow-llm-training/
├─ manifests/
│  ├─ pvc-datasets.yaml                 # 데이터셋/체크포인트 PVC (RWX)
│  ├─ pytorchjob-qlora.yaml             # 2노드 분산 PyTorchJob (QLoRA)
│  └─ configmap-train-scripts.yaml      # 트레이닝 스크립트/DeepSpeed 설정
├─ pipelines/
│  └─ data-prep-pipeline.py             # 데이터 준비(KFP) 파이프라인
├─ observability/
│  └─ tensorboard.yaml                  # TensorBoard 배포(Service 포함)
├─ katib/
│  └─ katib-experiment.yaml             # Katib HPO 예제(템플릿)
└─ README.md / README_ko.md
```

## 3. 스토리지/PVC 생성
```bash
kubectl apply -f kubeflow-llm-training/manifests/pvc-datasets.yaml
```
- llm-datasets-pvc: /mnt/datasets 마운트
- llm-checkpoints-pvc: /mnt/checkpoints 마운트

## 4. 데이터 준비 파이프라인(Kubeflow Pipelines)
- 예시 실행(파이프라인에서 컴포넌트는 내부적으로 datasets/transformers 설치):
```python
from kfp import Client
from kubeflow_llm_training.pipelines.data_prep_pipeline import data_prep_pipeline

client = Client()
client.create_run_from_pipeline_func(
    data_prep_pipeline,
    arguments={"hf_dataset": "Open-Orca/OpenOrca", "split": "train", "max_len": 4096},
)
```
- 결과: `/mnt/datasets/processed` 경로에 토크나이즈된 데이터셋 저장(DS 저장소와 파이프라인의 PVC 마운트 일치 필요)

## 5. QLoRA 학습 실행(PytorchJob)
1) 스크립트/설정 ConfigMap 적용
```bash
kubectl apply -f kubeflow-llm-training/manifests/configmap-train-scripts.yaml
```
2) PytorchJob 실행
```bash
kubectl apply -f kubeflow-llm-training/manifests/pytorchjob-qlora.yaml
```
- torchrun: `--nnodes=2`, `--nproc_per_node=8`
- 시작 시 필요한 패키지(transformers/datasets/peft/deepspeed/accelerate/langfuse/bitsandbytes/tensorboard) 설치
- 환경 변수로 HF 토큰 및 Langfuse 설정 주입

모니터링
```bash
kubectl get pytorchjob qwen3-30b-qlora-train -o yaml | less
kubectl logs -l role=master -f
kubectl logs -l role=worker -f
```

## 6. TensorBoard/Langfuse
- TensorBoard 배포 및 포트포워드
```bash
kubectl apply -f kubeflow-llm-training/observability/tensorboard.yaml
kubectl port-forward svc/tensorboard-llm 6006:80
```
- Langfuse: 트레이닝 스크립트가 `LANGFUSE_PUBLIC_KEY/SECRET_KEY/HOST`가 설정돼 있으면 로그/체크포인트 이벤트를 Langfuse로 전송합니다.

## 7. 리소스 최적화와 베스트 프랙티스
- 메모리 부족 시: `gradient_accumulation_steps` 증가, `max_seq_length` 축소, DeepSpeed ZeRO 단계 확인
- 네트워크: IB/RDMA 사용 시 NCCL 환경 변수 적절히 설정(NCCL_IB_DISABLE=0 등)
- 이미지 최적화: 시작 시 pip 설치 대신, 커스텀 이미지를 빌드해 종속성/스크립트 사전 포함 권장
- 체크포인트: `--save_steps`/`--save_total_limit`로 빈도/보존 관리, 저장소 용량 주시

## 8. 문제 해결
- Pod Pending: 노드 라벨(H100), GPU 리소스 가용성, PVC 바인딩 확인
- ImportError: 네트워크 격리/패키지 미러 확인, 사전 빌드 이미지 사용 고려
- NCCL 에러: 드라이버/펌웨어/다중 NIC 구성, NCCL_DEBUG=WARN/INFO로 진단

## 9. 추론 파이프라인과 연계
- 학습 산출물(어댑터/체크포인트)을 인퍼런스 스택에 연결:
  - KServe vLLM에서 모델 경로를 체크포인트로 변경하거나, 허깅페이스 허브/오브젝트 스토리지에 업로드 후 모델 식별자 교체
  - 성능 확인 후 트래픽 전환(blue/green 또는 점진 배포)

## 10. 참고 링크
- Kubeflow Training Operator: https://www.kubeflow.org/docs/components/training/
- PyTorchJob: https://www.kubeflow.org/docs/components/training/operators/pytorch/
- PEFT/QLoRA: https://github.com/huggingface/peft
- DeepSpeed: https://www.deepspeed.ai/
- NCCL: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/

