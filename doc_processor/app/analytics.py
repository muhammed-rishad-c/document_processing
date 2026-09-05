import json
import threading
from pathlib import Path
from datetime import datetime, timezone

ANALYTICS_LOG_PATH = Path(__file__).resolve().parent / "analytics_log.jsonl"
_write_lock = threading.Lock()


PRICE_PER_1K_INPUT_TOKENS = 0.0
PRICE_PER_1K_OUTPUT_TOKENS = 0.0


QUERY_STAGE_NAMES = {"query_embedding_ms", "vector_search_ms", "context_prep_ms", "llm_generation_ms"}
UPLOAD_STAGE_NAMES = {"document_processing_ms", "chunk_embedding_ms"}


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return round(
        (input_tokens / 1000) * PRICE_PER_1K_INPUT_TOKENS
        + (output_tokens / 1000) * PRICE_PER_1K_OUTPUT_TOKENS,
        6,
    )


def log_request(
    *,
    method: str,
    path: str,
    status_code: int,
    response_time_ms: float,
    input_tokens: int | None = None,
    context_tokens: int | None = None,
    output_tokens: int | None = None,
    stage_timings: dict | None = None,
) -> None:
    total_tokens = None
    estimated_cost = None
    if input_tokens is not None or output_tokens is not None:
        input_tokens = input_tokens or 0
        output_tokens = output_tokens or 0
        context_tokens = context_tokens or 0
        total_tokens = input_tokens + context_tokens + output_tokens
        estimated_cost = estimate_cost(input_tokens, output_tokens)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "path": path,
        "status_code": status_code,
        "response_time_ms": round(response_time_ms, 2),
        "input_tokens": input_tokens,
        "context_tokens": context_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": estimated_cost,
        "stage_timings_ms": stage_timings,
    }

    with _write_lock, open(ANALYTICS_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def read_all_entries() -> list[dict]:
    if not ANALYTICS_LOG_PATH.exists():
        return []
    entries = []
    with open(ANALYTICS_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def build_summary() -> dict:
    entries = read_all_entries()
    empty_tokens = {
        "total_input_tokens": 0,
        "total_context_tokens": 0,
        "total_output_tokens": 0,
        "total_tokens": 0,
        "total_estimated_cost_usd": 0.0,
        "requests_with_token_data": 0,
    }
    if not entries:
        return {
            "total_requests": 0,
            "requests_by_endpoint": {},
            "avg_response_time_ms": 0,
            "token_usage": empty_tokens,
            "stage_performance": {},
            "slowest_query_stage": None,
            "slowest_upload_stage": None,
        }

    total_requests = len(entries)
    total_latency = sum(e["response_time_ms"] for e in entries)

    by_endpoint: dict[str, dict] = {}
    for e in entries:
        key = f"{e['method']} {e['path']}"
        stats = by_endpoint.setdefault(key, {"count": 0, "_latency_sum": 0.0, "error_count": 0})
        stats["count"] += 1
        stats["_latency_sum"] += e["response_time_ms"]
        if e["status_code"] >= 400:
            stats["error_count"] += 1
    for stats in by_endpoint.values():
        stats["avg_latency_ms"] = round(stats.pop("_latency_sum") / stats["count"], 2)

    token_entries = [e for e in entries if e.get("total_tokens") is not None]
    token_usage = {
        "total_input_tokens": sum(e["input_tokens"] or 0 for e in token_entries),
        "total_context_tokens": sum(e["context_tokens"] or 0 for e in token_entries),
        "total_output_tokens": sum(e["output_tokens"] or 0 for e in token_entries),
        "total_estimated_cost_usd": round(sum(e["estimated_cost_usd"] or 0.0 for e in token_entries), 6),
        "requests_with_token_data": len(token_entries),
    }
    token_usage["total_tokens"] = (
        token_usage["total_input_tokens"] + token_usage["total_context_tokens"] + token_usage["total_output_tokens"]
    )

    # --- per-stage aggregation ---
    stage_values: dict[str, list[float]] = {}
    for e in entries:
        stages = e.get("stage_timings_ms")
        if not stages:
            continue
        for stage_name, ms in stages.items():
            stage_values.setdefault(stage_name, []).append(ms)

    stage_performance = {
        stage_name: {
            "avg_ms": round(sum(values) / len(values), 2),
            "count": len(values),
        }
        for stage_name, values in stage_values.items()
    }

    slowest_query_stage = max(
        (s for s in stage_performance if s in QUERY_STAGE_NAMES),
        key=lambda s: stage_performance[s]["avg_ms"],
        default=None,
    )
    slowest_upload_stage = max(
        (s for s in stage_performance if s in UPLOAD_STAGE_NAMES),
        key=lambda s: stage_performance[s]["avg_ms"],
        default=None,
    )

    return {
        "total_requests": total_requests,
        "requests_by_endpoint": by_endpoint,
        "avg_response_time_ms": round(total_latency / total_requests, 2),
        "token_usage": token_usage,
        "stage_performance": stage_performance,
        "slowest_query_stage": slowest_query_stage,
        "slowest_upload_stage": slowest_upload_stage,
    }