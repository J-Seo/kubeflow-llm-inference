# Kubeflow 개요와 PyTorchJob YAML 상세 가이드 (한국어)

이 문서는 Kubeflow와 Kubeflow Training Operator(PytorchJob)를 처음 접하는 한국어 사용자를 위해, 개념부터 실무 YAML 작성까지 단계별로 설명합니다. 특히 30B 파라미터급 모델(예: Qwen3-30B)의 "풀 파인튜닝(Full Fine-tuning)"에 초점을 맞춥니다.

---

## 1. Kubeflow 한눈에 보기

### 1.1 Kubeflow란?
- 목적: Kubernetes 위에서 머신러닝(ML) 워크로드(학습, 서빙, 파이프라인)를 일관되고 확장 가능하게 운영
- 왜 필요한가: 컨테이너로 랩핑된 ML 잡을 스케줄링/복구/확장/관측하며, 팀/환경 간 재현성 향상

### 1.2 핵심 컴포넌트
- Pipelines: 데이터 준비, 학습, 평가, 배포까지 이어지는 워크플로우 자동화
- Training Operator: 분산 학습 잡(PytorchJob, TFJob 등) 생명주기 관리
- Serving(KServe 등): 학습된 모델의 추론 배포/스케일링
- Add-ons: Katib(HPO), KFServing(KServe), Metadata, Notebooks 등

### 1.3 Kubeflow Training Operator 동작 방식
- 사용자: 학습 잡을 CR(Custom Resource, 예: PyTorchJob)로 제출
- Operator: CR을 감시하고, 해당 스펙에 맞춰 파드/서비스 생성 및 상태 관리
- 이점: 분산 학습 토폴로지(마스터/워커), 실패 재시도, 종료 정책 등을 선언적으로 설정

### 1.4 PyTorchJob 개념과 생명주기
- 구성: Master(=Chief) 1개 + Worker N개 (필요 시 Elastic/PS 전략 등)
- 통신: torchrun 또는 rendezvous(c10d) 기반의 DDP 통신
- 생명주기: Submitted → Creating → Running(각 Replica) → Succeeded/Failed → Clean-up(옵션)

### 1.5 Kubernetes 리소스와의 통합
- PVC(PersistentVolumeClaim): 데이터셋, 체크포인트 등의 영속 스토리지 공유
- Secret/ConfigMap: 토큰, 자격증명, 설정/스크립트 전달
- Service: 파드 검색/통신(특히 Master 서비스 DNS가 rendezvous에 활용됨)

---

## 2. PyTorchJob YAML 상세 가이드
아래는 풀 파인튜닝(Full FT)을 위한 PyTorchJob 스켈레톤입니다(주석 포함). 2노드(각 8×H100)에서 16GPU 학습을 가정합니다.

```yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: qwen3-30b-fullft
  labels:
    app: llm-training
spec:
  runPolicy:
    cleanPodPolicy: None   # 완료 후 파드 보존(None)/정리(All) 등 정책
  pytorchReplicaSpecs:
    Master:                # 마스터(1개) — rendezvous 및 전체 잡 오케스트레이션
      replicas: 1
      restartPolicy: OnFailure
      template:
        metadata:
          labels:
            role: master
        spec:
          # (GPU 노드 스케줄링) H100 노드에만 올라가도록 지정
          nodeSelector:
            kubernetes.io/arch: amd64
            nvidia.com/gpu.product: NVIDIA-H100
          tolerations:
            - key: nvidia.com/gpu
              operator: Exists
              effect: NoSchedule
          affinity:
            nodeAffinity:
              requiredDuringSchedulingIgnoredDuringExecution:
                nodeSelectorTerms:
                  - matchExpressions:
                      - key: nvidia.com/gpu.product
                        operator: In
                        values: ["NVIDIA-H100"]
          containers:
            - name: trainer
              image: <your-registry>/pytorch-fullft:latest  # 모든 의존성 포함 이미지 권장
              imagePullPolicy: IfNotPresent
              command: ["bash", "-lc"]
              args:
                - >-
                  # torchrun: 2노드 × 노드당 8 GPU = 16프로세스
                  torchrun --nproc_per_node=8 --nnodes=2 --rdzv_backend=c10d
                  --rdzv_endpoint=$(PYTORCH_JOB_MASTER_SERVICE_HOST):29400
                  /workspace/train/run_full_finetune.py
                  --model_name Qwen/Qwen3-30B-A3B-Thinking-2507-FP8
                  --dataset_dir /mnt/datasets/processed
                  --output_dir /mnt/checkpoints/qwen3-30b-fullft
                  --per_device_train_batch_size 1
                  --gradient_accumulation_steps 4
                  --learning_rate 5e-5
                  --num_train_epochs 2
                  --max_seq_length 4096
                  --bf16 True
                  --deepspeed /workspace/train/configs/ds_full.json
                  --logging_steps 10 --save_steps 500
              env:
                - name: HUGGING_FACE_HUB_TOKEN     # HF 모델/데이터 접근용
                  valueFrom:
                    secretKeyRef:
                      name: hf-token
                      key: HUGGING_FACE_HUB_TOKEN
                - name: NCCL_DEBUG                 # 통신 이슈 디버깅
                  value: WARN
                - name: NCCL_IB_DISABLE
                  value: "0"
                - name: NCCL_P2P_DISABLE
                  value: "0"
                - name: LANGFUSE_PUBLIC_KEY        # (선택) 관측용
                  valueFrom:
                    secretKeyRef:
                      name: langfuse-secrets
                      key: LANGFUSE_PUBLIC_KEY
                - name: LANGFUSE_SECRET_KEY
                  valueFrom:
                    secretKeyRef:
                      name: langfuse-secrets
                      key: LANGFUSE_SECRET_KEY
                - name: LANGFUSE_HOST
                  valueFrom:
                    secretKeyRef:
                      name: langfuse-secrets
                      key: LANGFUSE_HOST
              resources:
                limits:
                  nvidia.com/gpu: 8  # 노드당 8 GPU 사용
                  cpu: "16"
                  memory: 128Gi
                requests:
                  nvidia.com/gpu: 8
                  cpu: "8"
                  memory: 96Gi
              volumeMounts:
                - name: datasets
                  mountPath: /mnt/datasets
                - name: checkpoints
                  mountPath: /mnt/checkpoints
                - name: dshm
                  mountPath: /dev/shm
          volumes:
            - name: datasets
              persistentVolumeClaim:
                claimName: llm-datasets-pvc    # RWX 권장
            - name: checkpoints
              persistentVolumeClaim:
                claimName: llm-checkpoints-pvc # RWX 권장
            - name: dshm
              emptyDir:
                medium: Memory
                sizeLimit: 64Gi

    Worker:                # 워커(1개) — 2노드 구성을 위해 1리카 필요
      replicas: 1
      restartPolicy: OnFailure
      template:
        metadata:
          labels:
            role: worker
        spec:
          nodeSelector:
            kubernetes.io/arch: amd64
            nvidia.com/gpu.product: NVIDIA-H100
          tolerations:
            - key: nvidia.com/gpu
              operator: Exists
              effect: NoSchedule
          affinity:
            nodeAffinity:
              requiredDuringSchedulingIgnoredDuringExecution:
                nodeSelectorTerms:
                  - matchExpressions:
                      - key: nvidia.com/gpu.product
                        operator: In
                        values: ["NVIDIA-H100"]
          containers:
            - name: trainer
              image: <your-registry>/pytorch-fullft:latest
              imagePullPolicy: IfNotPresent
              command: ["bash", "-lc"]
              args:
                - >-
                  torchrun --nproc_per_node=8 --nnodes=2 --rdzv_backend=c10d
                  --rdzv_endpoint=$(PYTORCH_JOB_MASTER_SERVICE_HOST):29400
                  /workspace/train/run_full_finetune.py
                  --model_name Qwen/Qwen3-30B-A3B-Thinking-2507-FP8
                  --dataset_dir /mnt/datasets/processed
                  --output_dir /mnt/checkpoints/qwen3-30b-fullft
                  --per_device_train_batch_size 1
                  --gradient_accumulation_steps 4
                  --learning_rate 5e-5
                  --num_train_epochs 2
                  --max_seq_length 4096
                  --bf16 True
                  --deepspeed /workspace/train/configs/ds_full.json
                  --logging_steps 10 --save_steps 500
              env:
                - name: HUGGING_FACE_HUB_TOKEN
                  valueFrom:
                    secretKeyRef:
                      name: hf-token
                      key: HUGGING_FACE_HUB_TOKEN
                - name: NCCL_DEBUG
                  value: WARN
                - name: NCCL_IB_DISABLE
                  value: "0"
                - name: NCCL_P2P_DISABLE
                  value: "0"
              resources:
                limits:
                  nvidia.com/gpu: 8
                  cpu: "16"
                  memory: 128Gi
                requests:
                  nvidia.com/gpu: 8
                  cpu: "8"
                  memory: 96Gi
              volumeMounts:
                - name: datasets
                  mountPath: /mnt/datasets
                - name: checkpoints
                  mountPath: /mnt/checkpoints
                - name: dshm
                  mountPath: /dev/shm
          volumes:
            - name: datasets
              persistentVolumeClaim:
                claimName: llm-datasets-pvc
            - name: checkpoints
              persistentVolumeClaim:
                claimName: llm-checkpoints-pvc
            - name: dshm
              emptyDir:
                medium: Memory
                sizeLimit: 64Gi
```

### 2.1 섹션별 해설
- metadata: 잡 이름/라벨. 관측/관리 도구에서 필터링 기준으로 사용
- runPolicy.cleanPodPolicy: 실패 조사/재실행을 위해 파드 보존이 유용할 수 있음
- pytorchReplicaSpecs: Master와 Worker의 파드 템플릿 정의
- nodeSelector/tolerations/affinity: H100 노드에만 스케줄되도록 보장(클러스터 라벨 확인 필수)
- resources.requests/limits: GPU/CPU/메모리 예약 및 상한. 스케줄링과 성능 안정성에 영향
- volumeMounts/volumes: PVC와의 연결. 데이터/체크포인트 경로 일관성 유지
- env: HF 토큰/통신 파라미터/NCCL 및 (선택)Langfuse 설정
- args: torchrun/학습 스크립트 인자. rendezvous endpoint는 Master 서비스 DNS 사용

---

## 3. 풀 파인튜닝(Full FT) 포커스

### 3.1 리소스 요구사항(30B 모델)
- 메모리: 노드당 GPU 8장 기준, GPU당 80GB(예: H100 80GB) 권장
- CPU/시스템 메모리: 데이터 로딩/샘플링/로깅 여유 고려(예: pod당 16 vCPU/128GiB)
- 스토리지: 체크포인트 용량 매우 큼(수백 GB~ 수 TB). 보존 정책/주기 설정 중요

### 3.2 GPU 메모리/성능 고려
- 시퀀스 길이(max_seq_length), 마이크로배치(per_device_train_batch_size), GA(gradient_accumulation_steps)의 균형 조정
- 활성화 체크포인팅(activation checkpointing), ZeRO 최적화(DeepSpeed) 활용
- 통신: IB/RDMA 권장, NCCL 환경변수 튜닝(NCCL_DEBUG, IB_DISABLE, P2P_DISABLE 등)

### 3.3 DeepSpeed(Full FT) 구성 예시
`ds_full.json`(예시) — ConfigMap 혹은 이미지에 포함
```json
{
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 4,
  "bf16": {"enabled": true},
  "zero_optimization": {
    "stage": 2,
    "overlap_comm": true
  },
  "steps_per_print": 100,
  "wall_clock_breakdown": false
}
```
- stage: 2(메모리 절약), 필요 시 3으로 올리되 통신 오버헤드 고려
- bf16: H100에서는 bf16 혼합정밀 일반적. FP8 실험 시 프레임워크/TE 호환성 확인

---

## 4. 실전 패턴과 베스트 프랙티스
- 이미지 고정: 의존성/버전 고정으로 재현성 확보(pip install on-the-fly 지양)
- PVC 전략: 데이터/체크포인트는 RWX 스토리지. IOPS/Throughput 고려, 정기 정리
- 로그/모니터링: TensorBoard 디렉터리 분리, Prometheus/ELK 연동 고려
- 노드 스케줄링: Anti-Affinity로 동일 노드 과밀 방지, 우선순위 클래스 적용 검토
- 실패 대응: restartPolicy/RunPolicy와 경량 샘플로 드라이런 후 대규모 학습 수행

---

## 5. 문제 해결 가이드(YAML 관련)
- Pending(스케줄 안 됨):
  - nodeSelector 라벨/값 확인(H100 라벨 존재 여부), GPU 자원 가용성, PVC 바인딩
- ImagePullBackOff:
  - 레지스트리 접근 권한/시크릿, 이미지 태그/경로 오타
- CrashLoopBackOff:
  - command/args 경로/오타, 파이썬 의존성 누락(이미지 재빌드 권장)
- NCCL 오류:
  - 파드 간 통신 경로/보안그룹/서브넷, IB/RDMA 모듈, NCCL_DEBUG=INFO로 로그 확인
- 성능 저하:
  - seq length/배치/GA 재조정, ZeRO 단계, 데이터 로더 워커 수, 스토리지/네트워크 병목 점검

---

## 6. 추가 학습 자료
- Kubeflow Training Operator: https://www.kubeflow.org/docs/components/training/
- PyTorchJob: https://www.kubeflow.org/docs/components/training/operators/pytorch/
- Kubernetes 기본: https://kubernetes.io/ko/docs/home/
- DeepSpeed: https://www.deepspeed.ai/
- NCCL: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/

본 가이드를 기반으로 YAML을 점진적으로 확장/튜닝하시고, 작은 데이터와 짧은 학습으로 먼저 검증한 뒤 대규모 학습으로 전환하시길 권장드립니다.

