# -*- coding: utf-8 -*-
"""
Langfuse 연동 유틸리티(한국어 주석)
- get_langfuse(): 환경변수로부터 Langfuse 클라이언트를 초기화합니다.
- traced_observation(): with 문으로 관측 구간을 쉽게 감싸도록 도와줍니다.
주의: 환경변수 미설정 시 None을 반환하여 비파괴적으로 동작합니다.
"""
import os
from datetime import datetime
from contextlib import contextmanager

try:
    from langfuse import Langfuse
except Exception:
    Langfuse = None  # SDK 미설치/네트워크 문제 시 안전하게 비활성화


def get_langfuse():
    """환경변수(LANGFUSE_PUBLIC_KEY/SECRET_KEY/HOST)로 Langfuse 클라이언트를 초기화합니다.
    설정이 없으면 None을 반환하여 상위 코드가 관측 없이도 동작하도록 합니다.
    """
    if Langfuse is None:
        return None
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
    if not public_key or not secret_key:
        return None
    return Langfuse(public_key=public_key, secret_key=secret_key, host=host)


@contextmanager
def traced_observation(lf, name: str, metadata: dict | None = None):
    """Langfuse observation 컨텍스트 매니저.
    - lf가 None이면 관측 없이 통과합니다.
    - 블록 내 예외 발생 시 ERROR 레벨로 결과를 기록하고 다시 예외를 전파합니다.
    """
    if lf is None:
        yield None
        return
    obs = lf.observation(name=name, metadata=metadata or {})
    try:
        yield obs
        obs.end()
    except Exception as e:
        obs.update(output=str(e), level="ERROR")
        obs.end()
        raise

