"""Local Ollama access for report generation (no external API, no web access, no tools, no API key).

* Endpoint: http://localhost:11434 only (any other host is refused). The local HTTP API is called directly with the standard
  library, so no extra Python dependency is needed.
* Decoding is greedy and fixed: temperature 0, top_k 1, a fixed seed, an explicit context window (Ollama's default window would
  silently truncate long prompts) and a short output limit. Options are recorded with every response.
* Every successful response is cached by study id (invalidated if the request or the model digest changes), so an interrupted run
  never regenerates a completed case. A failed request is recorded as a failure; it is never replaced by fabricated text.
* The same model tag and options are used for G1 single-agent and the future G2 multi-agent systems.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

MODEL_TAG = "medgemma1.5:4b"
BASE_URL = "http://localhost:11434"
CHAT_URL = BASE_URL + "/api/chat"


@dataclass(frozen=True)
class GenerationConfig:
    model: str = MODEL_TAG
    temperature: float = 0.0
    top_k: int = 1                 # greedy decoding
    top_p: float = 1.0
    seed: int = 42
    num_predict: int = 400         # maximum new tokens (short Findings + Impression report)
    num_ctx: int = 8192            # explicit context window; the prompt is checked against it
    keep_alive: str = "30m"
    # The model sometimes continues after the IMPRESSION by echoing the input template; these markers (section titles of the INPUT message)
    # end generation there. This is a termination setting, not a change of the prompt wording.
    stop: tuple = ("A. AUTOMATED CLASSIFIER OUTPUT", "B. RETRIEVED EVIDENCE", "C. EVIDENCE SUMMARY")
    stream: bool = False
    tools: str = "none"
    web_access: bool = False
    external_api: bool = False

    def options(self) -> dict:
        return {"temperature": self.temperature, "top_k": self.top_k, "top_p": self.top_p, "seed": self.seed, "num_predict": self.num_predict, "num_ctx": self.num_ctx, "stop": list(self.stop)}

    def as_dict(self) -> dict:
        return asdict(self)


class OllamaTransientError(Exception):
    """Connection problem, timeout or server-side error: retried."""


class OllamaRequestError(Exception):
    """Request rejected (4xx) or malformed response: not retried."""


class OllamaClient:
    def __init__(self, base_url: str = BASE_URL, timeout: float = 600.0):
        host = urlparse(base_url).hostname
        if host not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError("only a local Ollama endpoint is allowed (no external API)")
        self.base, self.timeout = base_url.rstrip("/"), timeout

    def _call(self, path: str, payload: dict | None = None, timeout: float | None = None) -> dict:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(self.base + path, data=data, headers={"Content-Type": "application/json"}, method="POST" if data is not None else "GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", "ignore")[:300]
            except Exception:                                        # noqa: BLE001
                pass
            raise (OllamaTransientError if e.code >= 500 or e.code in (408, 429) else OllamaRequestError)(f"HTTP {e.code}: {body}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            raise OllamaTransientError(type(e).__name__) from None
        except json.JSONDecodeError:
            raise OllamaRequestError("response was not JSON") from None

    def version(self) -> str:
        return self._call("/api/version", timeout=10)["version"]

    def tags(self) -> list[dict]:
        return self._call("/api/tags", timeout=30).get("models", [])

    def show(self, model: str) -> dict:
        return self._call("/api/show", {"model": model}, timeout=60)

    def ps(self) -> list[dict]:
        return self._call("/api/ps", timeout=10).get("models", [])

    def chat(self, params: dict) -> dict:
        return self._call("/api/chat", params)


def installed_model(client, tag: str = MODEL_TAG) -> dict | None:
    return next((m for m in client.tags() if m.get("name") == tag or m.get("model") == tag), None)


def request_params(cfg: GenerationConfig, system: str, user: str) -> dict:
    return {"model": cfg.model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "stream": cfg.stream, "options": cfg.options(), "keep_alive": cfg.keep_alive}


def request_hash(params: dict, model_digest: str | None = None) -> str:
    return hashlib.sha256(json.dumps({"params": params, "model_digest": model_digest}, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


class ResponseCache:
    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, study_id: str) -> Path:
        return self.dir / f"{study_id}.json"

    def get(self, study_id: str, req_hash: str) -> dict | None:
        p = self.path(study_id)
        if not p.exists():
            return None
        d = json.loads(p.read_text(encoding="utf-8"))
        return d if d.get("request_hash") == req_hash and d.get("status") == "success" else None

    def put(self, study_id: str, record: dict) -> None:
        if record.get("status") != "success":
            raise ValueError("only successful responses are cached")
        self.path(study_id).write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")

    def completed(self, req_hashes: dict[str, str]) -> set[str]:
        return {s for s, h in req_hashes.items() if self.get(s, h) is not None}


def _record(resp: dict, params: dict, req_hash: str, attempts: int, seconds: float) -> dict:
    msg = resp.get("message") or {}
    if not isinstance(msg.get("content"), str) or not msg["content"].strip():
        raise OllamaRequestError("empty model response")
    prompt_tokens = resp.get("prompt_eval_count")
    return {"status": "success", "request_hash": req_hash, "text": msg["content"], "done_reason": resp.get("done_reason"), "model_returned": resp.get("model"), "prompt_tokens": prompt_tokens,
            "output_tokens": resp.get("eval_count"), "total_duration_ns": resp.get("total_duration"), "load_duration_ns": resp.get("load_duration"), "attempts": attempts,
            "wall_seconds": round(seconds, 2), "options": params["options"], "prompt_may_be_truncated": bool(prompt_tokens is not None and prompt_tokens >= params["options"]["num_ctx"] - 1),
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def generate_single(client, cfg: GenerationConfig, system: str, user: str, model_digest: str | None = None, max_attempts: int = 5, sleep=time.sleep) -> dict:
    """One chat call with retries on transient failures. Returns a success record or a failure record (no fabricated text)."""
    params = request_params(cfg, system, user)
    rh = request_hash(params, model_digest)
    last: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        t0 = time.time()
        try:
            return _record(client.chat(params), params, rh, attempt, time.time() - t0)
        except OllamaTransientError as e:
            last = e
            if attempt < max_attempts:
                sleep(min(60.0, 2.0 ** attempt))
        except OllamaRequestError as e:
            last = e
            break
    return {"status": "failed", "error_class": type(last).__name__, "error": str(last)[:200], "attempts": attempt}


def hardware_record(client=None) -> dict:
    """Hardware / backend used for generation (read-only inspection)."""
    rec = {"os": platform.platform(), "machine": platform.machine(), "processor": platform.processor()}
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
        rec["nvidia_smi"] = out or None
    except Exception:                                                # noqa: BLE001
        rec["nvidia_smi"] = None
    if client is not None:
        try:
            rec["ollama_ps"] = [{k: m.get(k) for k in ("name", "size", "size_vram", "context_length")} for m in client.ps()]
        except Exception:                                            # noqa: BLE001
            rec["ollama_ps"] = None
    return rec
