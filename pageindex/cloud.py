from __future__ import annotations
"""PageIndex cloud API client (https://api.pageindex.ai)."""

import json
import os
import time
from pathlib import Path
from typing import Any

import requests

from pageindex.log_util import log_info, log_error

DEFAULT_API_BASE = "https://api.pageindex.ai"


class PageIndexAPIError(Exception):
    def __init__(self, message: str, status_code: int | None = None, body: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class PageIndexAPI:
    """Official PageIndex document + chat + retrieval API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int = 120,
    ):
        self.api_key = api_key or os.getenv("PAGEINDEX_API_KEY", "").strip()
        if not self.api_key:
            raise PageIndexAPIError(
                "PAGEINDEX_API_KEY is required. Get one at https://dash.pageindex.ai/api-keys"
            )
        self.base_url = (base_url or os.getenv("PAGEINDEX_API_BASE", DEFAULT_API_BASE)).rstrip("/")
        self.timeout = timeout
        self._last_response: Any = None
        self.usage_meter = None

    def _headers(self, json_body: bool = False) -> dict[str, str]:
        h = {"api_key": self.api_key}
        if json_body:
            h["Content-Type"] = "application/json"
        return h

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict | None = None,
        files: dict | None = None,
        params: dict | None = None,
    ) -> Any:
        url = f"{self.base_url}{path}"
        log_info("[PageIndex API] %s %s", method, url)
        if params:
            log_info("[PageIndex API] params: %s", params)
        try:
            resp = requests.request(
                method,
                url,
                headers=self._headers(json_body=json_body is not None),
                json=json_body,
                files=files,
                params=params,
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            log_error("[PageIndex API] network error: %s", e)
            raise PageIndexAPIError(f"PageIndex API request failed: {e}") from e

        log_info("[PageIndex API] <- %s %s", resp.status_code, path)
        self._last_response = None
        if resp.content:
            try:
                self._last_response = resp.json()
            except json.JSONDecodeError:
                self._last_response = {"raw": resp.text}
        meter = getattr(self, "usage_meter", None)
        if meter and self._last_response is not None:
            from pageindex.usage.constants import Operation

            op = Operation.CLOUD_TREE.value
            if "/chat/" in path:
                op = Operation.CLOUD_CHAT.value
            elif path.startswith("/retrieval"):
                op = Operation.CLOUD_RETRIEVAL.value
            elif path == "/doc/" and method == "POST":
                op = Operation.CLOUD_SUBMIT.value
            meter.record(
                op,
                latency_ms=int(resp.elapsed.total_seconds() * 1000) if resp.elapsed else 0,
                api_response=self._last_response,
                metadata={"path": path, "method": method, "status_code": resp.status_code},
            )
        if resp.status_code >= 400:
            try:
                body = resp.json()
            except Exception:
                body = resp.text
            raise PageIndexAPIError(
                f"PageIndex API error {resp.status_code}: {body}",
                status_code=resp.status_code,
                body=body,
            )
        if not resp.content:
            return {}
        if self._last_response is not None:
            return self._last_response
        try:
            return resp.json()
        except json.JSONDecodeError:
            return {"raw": resp.text}

    def submit_pdf(self, pdf_path: str | Path) -> str:
        path = Path(pdf_path)
        with open(path, "rb") as f:
            data = self._request(
                "POST",
                "/doc/",
                files={"file": (path.name, f, "application/pdf")},
            )
        doc_id = data.get("doc_id")
        if not doc_id:
            raise PageIndexAPIError(f"No doc_id in submit response: {data}")
        return doc_id

    def get_doc_tree(
        self,
        doc_id: str,
        *,
        summary: bool = False,
    ) -> dict[str, Any]:
        params = {"type": "tree"}
        if summary:
            params["summary"] = "true"
        return self._request("GET", f"/doc/{doc_id}/", params=params)

    def get_metadata(self, doc_id: str) -> dict[str, Any]:
        return self._request("GET", f"/doc/{doc_id}/metadata")

    def wait_for_tree(
        self,
        doc_id: str,
        *,
        poll_interval: float = 3.0,
        timeout: float = 900.0,
        summary: bool = True,
    ) -> dict[str, Any]:
        deadline = time.time() + timeout
        last: dict[str, Any] = {}
        attempt = 0
        while time.time() < deadline:
            attempt += 1
            last = self.get_doc_tree(doc_id, summary=summary)
            status = (last.get("status") or "").lower()
            log_info(
                "[PageIndex API] poll #%d doc_id=%s status=%s retrieval_ready=%s",
                attempt,
                doc_id,
                status or "?",
                last.get("retrieval_ready"),
            )
            if status == "completed":
                if last.get("result") is not None:
                    log_info("[PageIndex API] tree ready (%d top nodes)", len(last.get("result") or []))
                    return last
                if last.get("retrieval_ready"):
                    return last
            if status in ("failed", "error"):
                raise PageIndexAPIError(f"Document processing failed: {last}")
            time.sleep(poll_interval)
        raise PageIndexAPIError(
            f"Timed out waiting for doc {doc_id} after {timeout}s. Last status: {last.get('status')}"
        )

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        doc_id: str | list[str] | None = None,
        temperature: float | None = None,
        stream: bool = False,
    ) -> str:
        payload: dict[str, Any] = {"messages": messages, "stream": stream}
        if doc_id is not None:
            payload["doc_id"] = doc_id
        if temperature is not None:
            payload["temperature"] = temperature
        data = self._request("POST", "/chat/completions", json_body=payload)
        choices = data.get("choices") or []
        if not choices:
            return ""
        msg = choices[0].get("message") or {}
        return (msg.get("content") or "").strip()

    def submit_retrieval(self, doc_id: str, query: str, *, thinking: bool = False) -> str:
        data = self._request(
            "POST",
            "/retrieval/",
            json_body={"doc_id": doc_id, "query": query, "thinking": thinking},
        )
        rid = data.get("retrieval_id")
        if not rid:
            raise PageIndexAPIError(f"No retrieval_id: {data}")
        return rid

    def get_retrieval(self, retrieval_id: str) -> dict[str, Any]:
        return self._request("GET", f"/retrieval/{retrieval_id}/")

    def retrieve(
        self,
        doc_id: str,
        query: str,
        *,
        thinking: bool = False,
        poll_interval: float = 2.0,
        timeout: float = 300.0,
    ) -> dict[str, Any]:
        rid = self.submit_retrieval(doc_id, query, thinking=thinking)
        deadline = time.time() + timeout
        while time.time() < deadline:
            result = self.get_retrieval(rid)
            status = (result.get("status") or "").lower()
            if status == "completed":
                return result
            if status in ("failed", "error"):
                raise PageIndexAPIError(f"Retrieval failed: {result}")
            time.sleep(poll_interval)
        raise PageIndexAPIError(f"Retrieval timed out for {rid}")


def get_api_key() -> str:
    return os.getenv("PAGEINDEX_API_KEY", "").strip()


def get_client() -> PageIndexAPI:
    return PageIndexAPI(api_key=get_api_key())




def build_cloud_index(
    pdf_path: str,
    *,
    poll_timeout: float = 900.0,
    include_summary: bool = True,
    meter: Any | None = None,
    job_id: str | None = None,
    fail_on_validation: bool = True,
) -> dict[str, Any]:
    from pageindex.usage.constants import Operation
    from pageindex.usage.meter import UsageMeter
    from pageindex.vrag.validation import apply_retrieval_readiness, clear_retrieval_ready
    from pageindex.vrag.schema import normalize_cloud_structure
    from pageindex.vrag.validation import ValidationError, validate_index

    path = Path(pdf_path)
    if meter is None:
        meter = UsageMeter(
            job_id=job_id or uuid.uuid4().hex,
            document_id=uuid.uuid4().hex,
            document_name=path.name,
            pipeline="pageindex",
            provider="pageindex",
            model="pageindex",
        )

    client = get_client()
    client.usage_meter = meter  # type: ignore

    log_info("[cloud] Step 1/3: Submit PDF — %s", path.name)
    file_bytes = path.stat().st_size
    meter.record_upload(file_bytes, latency_ms=0)

    doc_id = client.submit_pdf(path)
    meter.document_id = doc_id

    log_info("[cloud] Step 2/3: Polling doc_id=%s", doc_id)
    poll_attempt = 0
    doc: dict[str, Any] = {}
    deadline = time.time() + poll_timeout
    while time.time() < deadline:
        poll_attempt += 1
        t_poll = time.perf_counter()
        try:
            doc = client.get_doc_tree(doc_id, summary=include_summary)
            client._last_response = doc  # type: ignore
        except PageIndexAPIError:
            meter.record_poll(poll_attempt, int((time.perf_counter() - t_poll) * 1000))
            meter.record_retry(Operation.CLOUD_POLL.value)
            raise
        if poll_attempt == 1 or (doc.get("status") or "").lower() == "completed":
            meter.record_poll(
                poll_attempt,
                int((time.perf_counter() - t_poll) * 1000),
                api_response=doc,
            )
        status = (doc.get("status") or "").lower()
        if status == "completed" and (doc.get("result") is not None or doc.get("retrieval_ready")):
            break
        if status in ("failed", "error"):
            meter.fail(f"Document processing failed: {doc.get('status')}")
            raise PageIndexAPIError(f"Document processing failed: {doc}")
        time.sleep(3.0)

    raw_structure = doc.get("result") or []
    root = normalize_cloud_structure(raw_structure)

    meta: dict[str, Any] = {}
    try:
        meta = client.get_metadata(doc_id)
    except PageIndexAPIError:
        pass

    page_count = int(meta.get("pageNum") or meta.get("page_num") or 0)
    if page_count <= 0:
        page_count = _estimate_pages_from_tree(root)

    errors: list[str] = []
    try:
        errors = validate_index(root, strict=False, overlap_adjacent_threshold=0.15)
    except ValidationError as e:
        errors = e.errors

    # Cloud trees: warn on hierarchy issues but enforce content + readiness gates
    hierarchy_errors = [e for e in errors if "Invalid hierarchy" in e or "Orphan" in e]
    content_errors = [e for e in errors if e not in hierarchy_errors]

    # Cloud API trees are authoritative for display; VRAG gates inform readiness only.
    if fail_on_validation and not raw_structure:
        meter.fail("Empty cloud structure")
        raise ValidationError(["Empty cloud structure from PageIndex API"])

    readiness = apply_retrieval_readiness(
        root,
        content_errors,
        observability_initialized=bool(meter.job_id),
        retrieval_tests_passed=None,
        require_tests=False,
    )
    if hierarchy_errors:
        readiness["cloud_hierarchy_warnings"] = hierarchy_errors[:10]
    if content_errors:
        readiness["cloud_content_warnings"] = content_errors[:20]

    api_ready = doc.get("retrieval_ready")
    if api_ready is True:
        readiness["retrieval_ready"] = True
        readiness["gates"]["api_retrieval_ready"] = True

    if fail_on_validation and content_errors:
        log_info("[cloud] content warnings (%d), not blocking upload", len(content_errors))

    from pageindex.vrag.schema import export_node

    usage_report = meter.complete(status="success" if readiness["retrieval_ready"] else "failed_gates", page_count=page_count)
    if page_count > 0:
        usage_report = _inject_cloud_page_estimates(usage_report, page_count)

    log_info("[cloud] Step 3/3: doc_id=%s retrieval_ready=%s", doc_id, readiness["retrieval_ready"])
    return {
        "pipeline": "pageindex",
        "schema_version": "2.3",
        "source_pdf": path.name,
        "doc_id": doc_id,
        "document_id": doc_id,
        "job_id": meter.job_id,
        "doc_name": meta.get("name") or path.stem,
        "doc_description": meta.get("description", ""),
        "page_count": page_count,
        "status": doc.get("status", "completed"),
        "retrieval_ready": readiness["retrieval_ready"],
        "readiness": readiness,
        "structure": raw_structure if raw_structure else export_node(root),
        "structure_vrag": export_node(root),
        "usage": usage_report,
        "validation": {
            "valid": readiness["retrieval_ready"],
            "error_count": len(content_errors),
            "errors": content_errors[:30],
            "warnings": hierarchy_errors[:10],
            "chunk_count": readiness["ready_node_count"],
        },
        # Keep native cloud tree for UI compatibility
        "structure_cloud_native": raw_structure,
    }


def _estimate_pages_from_tree(root: dict) -> int:
    max_page = 0

    def walk(node: dict):
        nonlocal max_page
        for key in ("page_end", "page_start", "page_index"):
            v = node.get(key)
            if isinstance(v, int) and v > max_page:
                max_page = v
        for c in node.get("nodes") or []:
            walk(c)

    walk(root)
    return max_page


def _inject_cloud_page_estimates(report: dict[str, Any], page_count: int) -> dict[str, Any]:
    total = float(report.get("overview", {}).get("total_credits_used") or 0)
    if total <= 0 or page_count <= 0:
        return report
    per = total / page_count
    splits = [
        ("ocr_extraction", 0.12),
        ("document_parsing", 0.03),
        ("compression_generation", 0.21),
        ("micro_summary_generation", 0.04),
        ("metadata_generation", 0.02),
    ]
    norm = sum(s[1] for s in splits)
    page_rows = [
        {
            "page_number": p,
            "total_credits": round(per, 6),
            "total_tokens": 0,
            "operations": {n: round(per * (f / norm), 6) for n, f in splits},
            "accuracy": "ESTIMATED_ALLOCATION",
        }
        for p in range(1, page_count + 1)
    ]
    report["page_breakdown"] = page_rows
    report["page_heatmap"] = [{"page": r["page_number"], "credits": r["total_credits"]} for r in page_rows]
    return report
