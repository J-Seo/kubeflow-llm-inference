# Kubeflow 기반 대규모 언어모델 "포스트 트레이닝(미세튜닝)" 가이드 (한국어)

본 문서는 `kubeflow-llm-training/` 디렉터리를 사용하여 "사전학습(pre-training) 없이" 이미 공개/배포된 대규모 언어모델(예: Qwen/Qwen3-30B-A3B-Thinking-2507-FP8)을 2개 노드(각 8×H100, 총 16 GPU) 환경에서 "포스트 트레이닝(미세튜닝, fine-tuning)" 하는 방법을 설명합니다. 즉, 여기서는 거대 코퍼스로부터 처음부터 모델을 학습하는 "프리트레이닝"을 다루지 않습니다.

시나리오 환경 참고: 본 가이드는 제공된 scenario.pdf의 클라우드 서비스 배포 예시를 기준으로 작성되었으며, 실제 환경의 라벨/스토리지/네트워크 정책에 맞춰 일부 값(라벨 키/StorageClass/이미지 레지스트리/Ingress 등)을 조정해야 합니다.

## 1. 전제 조건(시나리오 환경 반영)
- Kubeflow Training Operator(PytorchJob CRD) 설치 상태
- GPU 노드 2개(각 H100 8장), 노드 라벨은 H100 식별 가능하도록 설정(예: `nvidia.com/gpu.product: NVIDIA-H100`)
- RWX 스토리지 클래스(NFS/CSI 등) — 데이터셋/체크포인트 공유(PVC 이름과 StorageClass는 시나리오 환경에 맞춤)
- Hugging Face 토큰 시크릿(`hf-token`) 존재
- (선택) Langfuse 관측 서버(`langfuse-secrets`) — 실시간 로깅/분석
- (네트워크) 아웃바운드 제한 시 프록시/미러 레지스트리 준비, 이미지 풀 비밀정보(ImagePullSecret) 필요 시 추가

## 2. 디렉터리 개요
```
kubeflow-llm-training/
├─ manifests/
│  ├─ pvc-datasets.yaml                 # 데이터셋/체크포인트 PVC (RWX)
│  ├─ pytorchjob-qlora.yaml             # 2노드 분산 PyTorchJob (QLoRA 기반 미세튜닝)
│  └─ configmap-train-scripts.yaml      # 트레이닝 스크립트(QLoRA/Full FT) + DeepSpeed 설정
├─ pipelines/
│  └─ data-prep-pipeline.py             # 데이터 준비(KFP) 파이프라인(토크나이즈/저장)
├─ observability/
│  └─ tensorboard.yaml                  # TensorBoard 배포(Service 포함)
├─ katib/
│  └─ katib-experiment.yaml             # Katib HPO 예시(템플릿)
└─ README.md / README_ko.md
```

## 3. 스토리지/PVC 생성(공유 저장)
```bash
kubectl apply -f kubeflow-llm-training/manifests/pvc-datasets.yaml
```
- llm-datasets-pvc → /mnt/datasets
- llm-checkpoints-pvc → /mnt/checkpoints
- scenario.pdf의 스토리지 클래스명이 `nfs-client`와 다르면 해당 값으로 변경하세요.

## 4. 데이터 준비(포스트 트레이닝용)
포스트 트레이닝은 모델이 이미 학습한 언어능력을 "특정 태스크/도메인"에 맞게 조정하는 과정입니다. 일반적으로 "지도 미세튜닝(SFT)" 형식의 데이터(Instruction/Response 쌍)를 사용합니다.

- 예시 실행(KFP 파이프라인):
```python
from kfp import Client
from kubeflow_llm_training.pipelines.data_prep_pipeline import data_prep_pipeline

client = Client()
client.create_run_from_pipeline_func(
    data_prep_pipeline,
    arguments={"hf_dataset": "Open-Orca/OpenOrca", "split": "train", "max_len": 4096},
)
```
- 출력: `/mnt/datasets/processed` 에 토크나이즈 저장(파이프라인과 학습 Job이 동일 PVC를 마운트해야 함)
- 데이터 팁(30B+ 포스트 트레이닝):
  - 품질 좋은 Instruction/Response가 핵심(노이즈/중복 제거)
  - max_len은 메모리/과금 영향이 큼(필요 최소로 시작, 점진 확대)

## 5. 포스트 트레이닝 실행(QLoRA 예시)
- ConfigMap(스크립트/설정) 적용
```bash
kubectl apply -f kubeflow-llm-training/manifests/configmap-train-scripts.yaml
```
- PyTorchJob 실행(QLoRA)
```bash
kubectl apply -f kubeflow-llm-training/manifests/pytorchjob-qlora.yaml
```
주요 특징(시나리오 환경 반영):
- torchrun: `--nnodes=2`, `--nproc_per_node=8`, rendezvous는 MASTER_ADDR/MASTER_PORT 활용(플랫폼 이식성↑)
- 경로 일관성: 데이터 `/mnt/datasets/processed`, 체크포인트 `/mnt/checkpoints`
- 의존성: transformers/datasets/peft/deepspeed/accelerate/langfuse/bitsandbytes/tensorboard
- 하이퍼파라미터(포스트 트레이닝 지향):
  - 학습률: 1e-4(어댑터) / 5e-5(Full FT 기준) 수준부터 스윕 권장
  - per_device_train_batch_size=1, gradient_accumulation_steps로 글로벌 배치 조정
  - bf16=True(H100 권장), max_seq_length는 4096에서 시작(리소스 따라 조정)

모니터링
```bash
kubectl get pytorchjob qwen3-30b-qlora-train -o yaml | less
kubectl logs -l role=master -f
kubectl logs -l role=worker -f
```

## 6. 대규모(30B+) 포스트 트레이닝 고려사항
- 메모리/최적화
  - ZeRO(Stage-2/3)로 옵티마이저/그래디언트 분산, activation checkpoint로 피크 메모리 절감
  - seq_length/GA/배치 균형 조정 (OOM 시 seq_length↓ or GA↑)
  - /dev/shm 여유(예: 64Gi) 및 데이터 로더 워커 수(num_workers) 조정
- 분산 구성
  - 본 리포지토리의 PyTorchJob은 노드당 8 GPU, 총 16 GPU DDP 구성을 기본 가정
  - Megatron-LM 대안은 TP/PP 조합으로 16 GPU를 하나의 모델 인스턴스로 구성(별도 디렉터리 참조)
- 체크포인트 관리
  - 30B 전체 파라미터는 체크포인트가 매우 큼 → `--save_steps`/`--save_total_limit` 엄격 관리
  - 장기 보관은 오브젝트 스토리지/S3 업로드 병행 권장
  - 재시작(resume_from_checkpoint) 경로 일관성 유지
- 리소스 할당 패턴(포스트 트레이닝 중심)
  - Pod당 GPU 8장, CPU 16 vCPU, 메모리 128Gi(요구량/성능에 맞춰 조정)
  - Anti-Affinity/라벨/톨러레이션으로 노드 배치 제어(시나리오 라벨 규칙 반영)

## 7. TensorBoard/Langfuse(관측)
- TensorBoard 배포 및 포트포워드
```bash
kubectl apply -f kubeflow-llm-training/observability/tensorboard.yaml
kubectl port-forward svc/tensorboard-llm 6006:80
```
- Langfuse: `LANGFUSE_PUBLIC_KEY/SECRET_KEY/HOST` 설정 시 학습 로그/저장 이벤트를 기록

## 8. 평가/검증(포스트 트레이닝 품질)
- 홀드아웃 검증셋에서 Perplexity/정확도/루브릭 기반 점수 측정
- 소규모 프롬프트 세트로 전/후 성능 비교(A/B)
- 배포 전 vLLM(InferenceService)로 샘플 추론 검증(응답 품질/안정성/지연시간)
- 과적합 방지: 정규화/데이터 다양성, early stopping(루브릭 기반) 고려

## 9. 리소스 최적화와 베스트 프랙티스(포스트 트레이닝 시나리오)
- 이미지: 사전 빌드로 의존성 고정(런타임 pip install 최소화)
- 스토리지: PVC 용량/IOPS 관리, 체크포인트 주기적 정리/백업
- 네트워크: IB/RDMA 권장, NCCL_* 튜닝(NCCL_DEBUG=WARN→INFO로 점검)
- 실패 대응: 작은 데이터/짧은 이터로 드라이런 → 스케일업

## 10. 문제 해결
- Pending: H100 라벨/톨러레이션/스토리지 클래스/Quota 확인
- ImagePullBackOff: 레지스트리 접근/이미지 태그/시크릿 확인
- NCCL 오류: 파드 간 네트워크/보안그룹/서브넷, 드라이버/펌웨어 점검
- ImportError: 제한된 네트워크 시 미러/사전 빌드 이미지 활용

## 11. 추론 파이프라인과 연계
- 산출물(어댑터/체크포인트)을 인퍼런스에 연결
  - KServe vLLM에서 체크포인트 경로 참조 또는 허브/오브젝트 스토리지 업로드 후 식별자 교체
  - 성능 검증 후 점진적 트래픽 전환(blue/green)

## 12. 참고 링크
- scenario.pdf(배포 환경 가이드)
- Kubeflow Training Operator: https://www.kubeflow.org/docs/components/training/
- PyTorchJob: https://www.kubeflow.org/docs/components/training/operators/pytorch/
- DeepSpeed: https://www.deepspeed.ai/
- NCCL: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/

