# Kubeflow 기반 대규모 언어모델 추론 배포 가이드 (한국어)

이 문서는 한국어 사용자(쿠버네티스/Kubeflow/Langfuse 경험이 적은 분)를 위해, Qwen/Qwen3-30B-A3B-Thinking-2507-FP8 모델을 2개 노드(각 8×H100, 총 16GPU)에서 추론용으로 배포하는 방법을 단계별로 설명합니다.

## 1. 사전 준비 사항(Prerequisites)
- Kubernetes 클러스터: GPU 워커 노드 2개(H100 8장 탑재), Kubernetes 1.27+ 권장
- Kubeflow 및 KServe 설치: InferenceService 리소스를 사용하기 위해 필요
- NVIDIA Device Plugin 설치: `nvidia.com/gpu` 리소스 인식
- 스토리지 클래스: RWX(ReadWriteMany) 지원 스토리지(NFS/CSI 등)
- 네트워크: IB/RDMA 권장(NCCL 성능)
- Hugging Face 액세스 토큰: 모델 다운로드용 (Secret으로 주입)
- (선택) Langfuse: 관측/트레이싱 UI 서버

참고 문서
- Kubernetes: https://kubernetes.io/ko/docs/home/
- Kubeflow: https://www.kubeflow.org/docs/
- KServe: https://kserve.github.io/website/
- Langfuse: https://langfuse.com/

## 2. 디렉터리 개요
```
kubeflow-llm-inference/
├─ kserve/
│  ├─ vllm/
│  │  ├─ inference-service.yaml   # vLLM 추론(2레플리카 × 8GPU)
│  │  ├─ hf-secret.yaml           # HF 토큰 시크릿(수정 필요)
│  │  └─ pvc-model.yaml           # 모델 캐시 PVC(RWX)
│  └─ tgi/
│     └─ inference-service.yaml   # TGI 예시(옵션)
├─ pipelines/
│  └─ batch-client-pipeline.py    # 배치 추론 KFP 파이프라인 예시
├─ clients/
│  ├─ query_vllm.py               # OpenAI API 호환 엔드포인트 호출 클라이언트
│  └─ langfuse_utils.py           # Langfuse 헬퍼
├─ ray-option/
│  ├─ raycluster.yaml             # KubeRay 클러스터
│  └─ vllm-ray-inference.yaml     # Ray 백엔드 vLLM 추론(16GPU 단일 인스턴스)
└─ README.md / README_ko.md
```

## 3. 배포 절차(권장: vLLM on KServe)
### 3.1 Hugging Face Secret 및 PVC 생성
- hf-secret.yaml에서 `REPLACE_WITH_YOUR_TOKEN`을 실제 토큰으로 바꿉니다.
```bash
kubectl apply -f kubeflow-llm-inference/kserve/vllm/hf-secret.yaml
kubectl apply -f kubeflow-llm-inference/kserve/vllm/pvc-model.yaml
```

### 3.2 vLLM InferenceService 배포
```bash
kubectl apply -f kubeflow-llm-inference/kserve/vllm/inference-service.yaml
kubectl get isvc vllm-qwen3-30b-fp8 -w
```
- 준비(Ready) 상태가 되면 URL을 확인합니다.
```bash
kubectl get isvc vllm-qwen3-30b-fp8 -o jsonpath='{.status.url}'
```
- OpenAI ChatCompletions 엔드포인트: `${URL}/v1/chat/completions`

### 3.3 Langfuse 설정(선택)
- Langfuse가 없다면, `observability/langfuse/`의 매니페스트로 배포할 수 있습니다(인퍼런스 README 참고).
- 클라이언트/파이프라인에서 다음 환경 변수를 설정해야 Langfuse에 기록됩니다.
  - `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`

## 4. 클라이언트 사용 예시
- 모델 엔드포인트 호출
```bash
export LANGFUSE_PUBLIC_KEY=... LANGFUSE_SECRET_KEY=... LANGFUSE_HOST=https://langfuse.example.com
python kubeflow-llm-inference/clients/query_vllm.py \
  --url "$(kubectl get isvc vllm-qwen3-30b-fp8 -o jsonpath='{.status.url}')/v1/chat/completions" \
  --model qwen3-30b-fp8 \
  --prompt "쿠버플로우가 무엇인지 세 문장으로 설명해줘."
```
- 예상 출력(일례):
```json
{
  "id": "chatcmpl-...",
  "choices": [ { "message": { "role": "assistant", "content": "..." } } ],
  "usage": { "prompt_tokens": 25, "completion_tokens": 128, "total_tokens": 153 }
}
```
- Elapsed: 2.34s 와 같은 응답 지연 시간 표시

## 5. Ray 기반(옵션)
- 단일 논리 16GPU vLLM 인스턴스를 원한다면 Ray 옵션 사용
```bash
kubectl apply -f kubeflow-llm-inference/ray-option/raycluster.yaml
kubectl apply -f kubeflow-llm-inference/ray-option/vllm-ray-inference.yaml
```
- Service로 8080(HTTP), 8000(메트릭)을 노출(인그레스/포트포워딩 구성 필요)

## 6. 모니터링 & 로깅
- Prometheus 스크랩 어노테이션이 포함되어 있으며, kube-prometheus-stack 사용 시 자동 수집 가능
- 메트릭 전용 Service: vLLM에 8000 포트 노출
- 로그 확인: `kubectl logs -l app=vllm-qwen3-30b-fp8 -f`
- Health 체크: readiness/liveness/startup 프로브 구성 완료

## 7. 자원/병렬화 설정
- KServe vLLM: 레플리카 2×(TP=8) → 총 16GPU 활용
- Ray vLLM: `--distributed-executor-backend=ray`, `--tensor-parallel-size=16`
- CPU/메모리/임시스토리지 요청/제한 설정 포함(환경에 맞게 조정)

## 8. 문제 해결(Troubleshooting)
- Pending(스케줄 불가):
  - 노드 라벨(nvidia.com/gpu.product) 확인, GPU 쿼터, PVC 바인딩 여부 점검
- OOM/메모리 부족:
  - `--max-num-batched-tokens`, `--max-model-len`, `--gpu-memory-utilization` 조정
- 네트워크/NCCL 문제:
  - IB/RDMA 드라이버 확인, `NCCL_*` 환경 변수(IB_DISABLE, P2P_DISABLE 등) 점검
- 응답 지연이 큰 경우:
  - 배치/시퀀스 길이 조절, TPS 모니터링, 노드 간 균형 확인
- Langfuse에 기록이 안 되는 경우:
  - LANGFUSE_* 환경변수와 Langfuse URL 접근성 확인, Ingress/TLS 설정 점검

## 9. 권장 사항
- PVC 용량/성능은 FP8 가중치 캐시를 감안해 넉넉히 설정(기본 3Ti)
- 노드간 Anti-Affinity로 레플리카 분산
- 운영 환경에서는 고정 버전 이미지 사용 및 자원 상한(Quota) 관리 권장

## 10. 추가 자료
- vLLM: https://docs.vllm.ai/
- Hugging Face TGI: https://github.com/huggingface/text-generation-inference
- KubeRay: https://ray-project.github.io/kuberay/
- Prometheus/Grafana: https://prometheus.io/, https://grafana.com/

