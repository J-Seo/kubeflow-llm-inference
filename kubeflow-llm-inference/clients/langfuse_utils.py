import os
from datetime import datetime
from contextlib import contextmanager

try:
    from langfuse import Langfuse
except Exception:
    Langfuse = None


def get_langfuse():
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

