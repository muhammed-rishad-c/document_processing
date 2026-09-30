import re
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import text

from .models import DocumentChunk, CompanyDataSource, Company, Document
from .service import count_token, chunk_text_parent_child
from .vector_store import store_chunk_vector, qdrant, COLLECTION_NAME
from qdrant_client.models import Filter, FieldCondition, MatchValue

_CHUNK_NAMESPACE = uuid.NAMESPACE_DNS

EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

def _flatten_extra_lines(extra: dict, prefix: str = "") -> list[str]:
    """Recursively flatten `extra` into readable lines. Postgres `extra` is
    always a flat dict, so this produces identical output to the old
    non-recursive version for every existing row — the recursion only ever
    fires for Mongo's nested objects/lists (Plan Phase 3, section 8.1)."""
    lines = []
    for key, value in (extra or {}).items():
        label = f"{prefix}{key.replace('_', ' ').capitalize()}"
        if isinstance(value, dict):
            lines += _flatten_extra_lines(value, prefix=f"{label} - ")
        elif isinstance(value, list):
            lines.append(f"{label}: " + ", ".join(str(v) for v in value))
        elif value is not None:
            lines.append(f"{label}: {value}")
    return lines

def _format_row_chunk_text(row: dict) -> str:
    """Generic row -> chunk text formatter. No changes needed when a new
    content_type or a new `extra` field shows up for any company — it just
    flows through. Shared by both the Postgres and Mongo sync paths."""
    lines = [row["title"], row["body"]]
    lines += _flatten_extra_lines(row.get("extra") or {})
    return "\n".join(lines)

def _to_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return ", ".join(t for t in (_to_text(v) for v in value) if t)
    if isinstance(value, dict):
        return "\n".join(_flatten_extra_lines(value))
    return str(value)

def _hybrid_row(doc: dict) -> dict:
    """Existing behavior: docs already in title/body/content_type/extra shape."""
    return {
        "id": str(doc["_id"]),
        "title": doc.get("title", ""),
        "body": doc.get("body", ""),
        "content_type": doc.get("content_type") or "general",
        "extra": doc.get("extra") or {},
    }

# ---- Layer 1: config-driven normalization ----------------------------------
_CONFIG_SYSTEM_KEYS = {"_id", "__v", "is_deleted", "updated_at", "created_at", "sample_tag"}
_SENSITIVE_RE = re.compile(
    r"(password|passwd|secret|token|api[_-]?key|salt|hash|salary|cost|internal|private|ssn)",
    re.IGNORECASE,
)
_ALLOWED_MODES = {"all_except_ignored", "only_listed"}
_MAX_LIST = 100
_MAX_NAME = 100

def _get_path(doc: dict, path: str):
    """'contacts.email' -> doc['contacts']['email']; None if any part is missing."""
    cur = doc
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur

def validate_field_config(config) -> dict:
    """Validate a config and return a cleaned copy. Raises ValueError."""
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")

    def _names(key: str) -> list[str]:
        value = config.get(key, [])
        if not isinstance(value, list):
            raise ValueError(f"{key} must be a list of field names")
        if len(value) > _MAX_LIST:
            raise ValueError(f"{key} has too many entries (max {_MAX_LIST})")
        out = []
        for item in value:
            if not isinstance(item, str) or not item.strip() or len(item) > _MAX_NAME:
                raise ValueError(f"{key} contains an invalid field name: {item!r}")
            out.append(item.strip())
        return out

    mode = config.get("mode", "all_except_ignored")
    if mode not in _ALLOWED_MODES:
        raise ValueError(f"mode must be one of {sorted(_ALLOWED_MODES)}")

    labels = config.get("labels", {})
    if not isinstance(labels, dict) or len(labels) > _MAX_LIST:
        raise ValueError("labels must be an object of field name -> label")
    for k, v in labels.items():
        if not isinstance(k, str) or not isinstance(v, str) or len(k) > _MAX_NAME or len(v) > _MAX_NAME:
            raise ValueError("labels must map short strings to short strings")

    ct_field = config.get("content_type_field", "content_type")
    if not isinstance(ct_field, str) or len(ct_field) > _MAX_NAME:
        raise ValueError("content_type_field must be a field name")

    return {
        "title_fields": _names("title_fields"),
        "body_fields": _names("body_fields"),
        "ignore_fields": _names("ignore_fields"),
        "include_fields": _names("include_fields"),
        "labels": dict(labels),
        "mode": mode,
        "content_type_field": ct_field,
    }

def _apply_field_config(doc: dict, config: dict) -> dict | None:
    """Mongo doc -> {title, body, content_type, extra} using the config.
    Returns None if no title and no body were found (caller falls back)."""
    title_fields = config.get("title_fields", [])
    body_fields = config.get("body_fields", [])
    include = set(config.get("include_fields", []))
    ignore = set(config.get("ignore_fields", [])) | _CONFIG_SYSTEM_KEYS
    labels = config.get("labels", {})
    only_listed = config.get("mode") == "only_listed"

    used: set[str] = set()

    title = ""
    for path in title_fields:
        value = _to_text(_get_path(doc, path))
        if value:
            title = value
            if "." not in path:
                used.add(path)
            break

    body_parts = []
    for path in body_fields:
        value = _to_text(_get_path(doc, path))
        if value:
            body_parts.append(value)
            if "." not in path:
                used.add(path)
    body = "\n".join(body_parts)

    if not title and not body:
        return None

    # Fields a person listed on purpose bypass the sensitive-name filter.
    explicit = {p.split(".")[0] for p in title_fields + body_fields} | include

    extra: dict = {}
    for key, value in doc.items():
        if key in used or key in ignore or value is None:
            continue
        if only_listed and key not in include:
            continue
        if _SENSITIVE_RE.search(key) and key not in explicit:
            continue
        extra[labels.get(key, key)] = value

    content_type_key = config.get("content_type_field", "content_type")
    return {
        "title": title,
        "body": body,
        "content_type": _to_text(doc.get(content_type_key)) or "general",
        "extra": extra,
    }

def _load_active_config(source) -> dict | None:
    """Config for this sync, or None to use the existing hybrid rules.
    Only an APPROVED, valid config is used."""
    if source.field_config_status != "approved" or not source.field_config:
        return None
    try:
        return validate_field_config(source.field_config)
    except ValueError as e:
        print(f"[mongo:sync] company={source.company_id} ignoring invalid config: {e}")
        return None

def _delete_row_chunks(db_session, company_id, row_pk, data_source_id) -> None:   
    
    existing_children = (
        db_session.query(DocumentChunk)
        .filter(
            DocumentChunk.company_id == company_id,
            DocumentChunk.data_source_id == data_source_id,        
            DocumentChunk.source_product_id == row_pk,
            DocumentChunk.is_parent == False,
        )
        .all()
    )
    for child in existing_children:
        qdrant.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(
                must=[FieldCondition(key="metadata.chunk_id", match=MatchValue(value=str(child.id)))]
            ),
        )
    db_session.query(DocumentChunk).filter(
        DocumentChunk.company_id == company_id,
        DocumentChunk.data_source_id == data_source_id,            
        DocumentChunk.source_product_id == row_pk,
    ).delete(synchronize_session=False)

def _chunk_uuid_for_row(company_id, row_pk) -> uuid.UUID:
    return uuid.uuid5(_CHUNK_NAMESPACE, f"product:{company_id}:{row_pk}")

def _get_or_create_db_source_document(db_session, company_id) -> uuid.UUID:
    """One shared Document per company for all its synced product data
    (source_type='db_source'), reused across every CompanyDataSource row
    that company has. Created lazily on first sync."""
    doc = (
        db_session.query(Document)
        .filter(Document.company_id == company_id, Document.source_type == "db_source")
        .first()
    )
    if doc is not None:
        return doc.id

    company = db_session.query(Company).filter(Company.id == company_id).first()
    if company is None:
        raise ValueError(f"Cannot create db_source Document: company {company_id} not found.")

    doc = Document(
        filename=f"{company.name} — synced product data",
        file_type="db_source",
        extracted_text="",
        stats={},
        structure=None,
        company_id=company_id,
        source_type="db_source",
    )
    db_session.add(doc)
    db_session.flush()  # assigns doc.id without committing yet
    return doc.id

def sync_company_data_source(db_session, source: CompanyDataSource) -> dict:
    if (source.source_type or "postgres") != "postgres":                      # ADDED: guard for future Mongo rows
        return {"rows_upserted": 0, "rows_deleted": 0, "duration_ms": None,
                "skipped": f"source_type={source.source_type}"}

    table = source.source_table
    since = source.last_synced_at or EPOCH

    document_id = _get_or_create_db_source_document(db_session, source.company_id)

    changed_rows = db_session.execute(
        text(f'SELECT * FROM "{table}" WHERE updated_at > :since'),
        {"since": since},
    ).mappings().all()

    rows_upserted = 0
    db_chunks_to_add = []
    vector_data = []
    synced_pks: set[str] = set()

    for row in changed_rows:
        row = dict(row)
        row_pk = str(row["id"])
        synced_pks.add(row_pk)

        chunk_text_full = _format_row_chunk_text(row)
        content_type = row.get("content_type")

        _delete_row_chunks(db_session, source.company_id, row_pk, source.id)   # CHANGED: passes source.id

        pieces = chunk_text_parent_child(chunk_text_full)

        local_parent_global_index = {}
        for piece in pieces:
            if not piece["is_parent"]:
                continue
            parent_uuid = uuid.uuid4()
            global_idx = parent_uuid.int % (2**31)
            local_parent_global_index[piece["chunk_index"]] = global_idx
            db_chunks_to_add.append(DocumentChunk(
                id=parent_uuid,
                document_id=document_id,
                chunk_index=global_idx,
                chunk_text=piece["chunk_text"],
                token_count=piece["token_count"],
                is_parent=True,
                parent_index=None,
                source_product_id=row_pk,
                company_id=source.company_id,
                content_type=content_type,
                data_source_id=source.id,                                      # ADDED
            ))

        for piece in pieces:
            if piece["is_parent"]:
                continue
            child_uuid = uuid.uuid4()
            global_parent_idx = local_parent_global_index[piece["parent_index"]]
            child_chunk = DocumentChunk(
                id=child_uuid,
                document_id=document_id,
                chunk_index=child_uuid.int % (2**31),
                chunk_text=piece["chunk_text"],
                token_count=piece["token_count"],
                is_parent=False,
                parent_index=global_parent_idx,
                source_product_id=row_pk,
                company_id=source.company_id,
                content_type=content_type,
                data_source_id=source.id,                                      # ADDED
            )
            db_chunks_to_add.append(child_chunk)
            vector_data.append({
                "point_id": child_uuid,
                "document_id": document_id,
                "company_id": source.company_id,
                "chunk_index": child_chunk.chunk_index,
                "chunk_text": piece["chunk_text"],
                "token_count": piece["token_count"],
                "parent_index": global_parent_idx,
                "is_parent": False,
                "embedding": None,
            })
        rows_upserted += 1

    if vector_data:
        from .vector_store import get_embeddings_batch
        texts = [v["chunk_text"] for v in vector_data]
        embeddings = get_embeddings_batch(texts)
        for v, emb in zip(vector_data, embeddings):
            v["embedding"] = emb

    if db_chunks_to_add:
        db_session.add_all(db_chunks_to_add)

    # --- Diff against all previously-synced PKs for this source to handle deletes ---
    current_pks = {
        str(r["id"])
        for r in db_session.execute(text(f'SELECT id FROM "{table}"')).mappings().all()
    }
    stale_rows = (
        db_session.query(DocumentChunk)
        .filter(
            DocumentChunk.company_id == source.company_id,
            DocumentChunk.data_source_id == source.id,                         # ADDED: only this source's chunks
            DocumentChunk.source_product_id.isnot(None),
        )
        .all()
    )
    stale_pks_to_delete = {
        row.source_product_id for row in stale_rows
        if row.source_product_id not in current_pks
    }
    rows_deleted = 0
    if stale_pks_to_delete:
        for pk in stale_pks_to_delete:
            _delete_row_chunks(db_session, source.company_id, pk, source.id)   # CHANGED: replaces the copied delete block
            rows_deleted += 1

    source.last_synced_at = datetime.now(timezone.utc)
    db_session.add(source)
    db_session.commit()

    if vector_data:
        store_chunk_vector(vector_data)

    return {
        "rows_upserted": rows_upserted,
        "rows_deleted": rows_deleted,
        "duration_ms": None,
    }

DB_SYNC_OVERLAP_SECONDS = 10

def sync_mongo_source(db_session, source: CompanyDataSource) -> dict:
    if (source.source_type or "postgres") != "mongo":
        return {"rows_upserted": 0, "rows_deleted": 0, "duration_ms": None,
                "skipped": f"source_type={source.source_type}"}

    from .mongo_client import get_mongo_db

    since = source.last_synced_at or EPOCH

    config = _load_active_config(source)
    config_version = source.field_config_version or 0
    if config is not None and (source.synced_config_version or 0) != config_version:
        since = EPOCH  

    sync_start = datetime.now(timezone.utc)
    watermark = sync_start - timedelta(seconds=DB_SYNC_OVERLAP_SECONDS)

    col = get_mongo_db()[source.collection_name]

    document_id = _get_or_create_db_source_document(db_session, source.company_id)

    changed_docs = list(col.find({"updated_at": {"$gt": since}}).sort("updated_at", 1))

    rows_upserted = 0
    rows_deleted = 0
    rows_by_config = 0
    rows_by_fallback = 0
    db_chunks_to_add = []
    vector_data = []

    for doc in changed_docs:
        row_pk = str(doc["_id"])

        # Every changed doc — including a soft-delete — starts by clearing
        # its old chunks. Same reasoning as the Postgres path: sync is
        # delete-then-insert, so it's safe to run unconditionally.
        _delete_row_chunks(db_session, source.company_id, row_pk, source.id)

        if bool(doc.get("is_deleted", False)):
            rows_deleted += 1
            continue

        row = _apply_field_config(doc, config) if config is not None else None
        if row is not None:
            rows_by_config += 1
        else:
            row = _hybrid_row(doc)
            rows_by_fallback += 1
        chunk_text_full = _format_row_chunk_text(row)
        content_type = row["content_type"]

        pieces = chunk_text_parent_child(chunk_text_full)

        local_parent_global_index = {}
        for piece in pieces:
            if not piece["is_parent"]:
                continue
            parent_uuid = uuid.uuid4()
            global_idx = parent_uuid.int % (2**31)
            local_parent_global_index[piece["chunk_index"]] = global_idx
            db_chunks_to_add.append(DocumentChunk(
                id=parent_uuid,
                document_id=document_id,
                chunk_index=global_idx,
                chunk_text=piece["chunk_text"],
                token_count=piece["token_count"],
                is_parent=True,
                parent_index=None,
                source_product_id=row_pk,
                company_id=source.company_id,
                content_type=content_type,
                data_source_id=source.id,
            ))

        for piece in pieces:
            if piece["is_parent"]:
                continue
            child_uuid = uuid.uuid4()
            global_parent_idx = local_parent_global_index[piece["parent_index"]]
            child_chunk = DocumentChunk(
                id=child_uuid,
                document_id=document_id,
                chunk_index=child_uuid.int % (2**31),
                chunk_text=piece["chunk_text"],
                token_count=piece["token_count"],
                is_parent=False,
                parent_index=global_parent_idx,
                source_product_id=row_pk,
                company_id=source.company_id,
                content_type=content_type,
                data_source_id=source.id,
            )
            db_chunks_to_add.append(child_chunk)
            vector_data.append({
                "point_id": child_uuid,
                "document_id": document_id,
                "company_id": source.company_id,
                "chunk_index": child_chunk.chunk_index,
                "chunk_text": piece["chunk_text"],
                "token_count": piece["token_count"],
                "parent_index": global_parent_idx,
                "is_parent": False,
                "embedding": None,
            })
        rows_upserted += 1

    if vector_data:
        from .vector_store import get_embeddings_batch
        texts = [v["chunk_text"] for v in vector_data]
        embeddings = get_embeddings_batch(texts)
        for v, emb in zip(vector_data, embeddings):
            v["embedding"] = emb

    if db_chunks_to_add:
        db_session.add_all(db_chunks_to_add)

    # --- Orphan scan: catches hard-deletes made directly in Mongo, ---
    # --- bypassing soft-delete. Mirrors the Postgres path's full scan. ---
    live_ids = {
        str(d["_id"]) for d in col.find({"is_deleted": {"$ne": True}}, {"_id": 1})
    }
    stale_rows = (
        db_session.query(DocumentChunk)
        .filter(
            DocumentChunk.company_id == source.company_id,
            DocumentChunk.data_source_id == source.id,
            DocumentChunk.source_product_id.isnot(None),
        )
        .all()
    )
    stale_pks_to_delete = {
        row.source_product_id for row in stale_rows
        if row.source_product_id not in live_ids
    }
    for pk in stale_pks_to_delete:
        _delete_row_chunks(db_session, source.company_id, pk, source.id)
        rows_deleted += 1

    source.last_synced_at = watermark
    source.last_run_at = sync_start
    if config is not None:
        source.synced_config_version = config_version
    source.last_status = "ok"
    source.last_error = None
    db_session.add(source)
    db_session.commit()

    if vector_data:
        store_chunk_vector(vector_data)

    return {
        "rows_upserted": rows_upserted,
        "rows_deleted": rows_deleted,
        "rows_by_config": rows_by_config,
        "rows_by_fallback": rows_by_fallback,
        "duration_ms": None,
    }
    
def preview_field_config(source, config: dict, limit: int = 5) -> list[dict]:
    from .mongo_client import get_mongo_db
    cleaned = validate_field_config(config)
    col = get_mongo_db()[source.collection_name]
    results = []
    for doc in col.find({"is_deleted": {"$ne": True}}).sort("_id", -1).limit(limit):
        row = _apply_field_config(doc, cleaned)
        layer = "config"
        if row is None:
            row = _hybrid_row(doc)
            layer = "fallback"
        results.append({
            "doc_id": str(doc["_id"]),
            "layer": layer,
            "embedded_text": _format_row_chunk_text(row),
        })
    return results

def _record_sync_status(db_session, source_id, status: str, error: str | None) -> None:
    try:
        db_session.execute(
            text("""
                UPDATE company_data_sources
                   SET last_status = :status, last_error = :err, last_run_at = now()
                 WHERE id = :id
            """),
            {"id": source_id, "status": status, "err": error},
        )
        db_session.commit()
    except Exception as e:
        db_session.rollback()
        print(f"[db_sync] could not record status for source={source_id}: {e}")

def sync_data_source(db_session, source: CompanyDataSource) -> dict:
    source_id = source.id
    is_mongo = (source.source_type or "postgres") == "mongo"
    try:
        summary = (
            sync_mongo_source(db_session, source)
            if is_mongo
            else sync_company_data_source(db_session, source)
        )
    except Exception as e:
        db_session.rollback()
        _record_sync_status(db_session, source_id, "error", str(e)[:1000])
        raise  # callers still handle/log it exactly as before
    if not is_mongo:
        _record_sync_status(db_session, source_id, "ok", None)  # Mongo path already does this itself
    return summary

LEASE_DURATION = timedelta(minutes=10)

def acquire_sync_lease(db_session, source: CompanyDataSource) -> bool:
    
    result = db_session.execute(
        text("""
            UPDATE company_data_sources
               SET sync_locked_until = now() + :lease
             WHERE id = :id
               AND (sync_locked_until IS NULL OR sync_locked_until < now())
        """),
        {"id": source.id, "lease": LEASE_DURATION},
    )
    db_session.commit()
    return result.rowcount == 1

def release_sync_lease(db_session, source: CompanyDataSource) -> None:
    db_session.execute(
        text("UPDATE company_data_sources SET sync_locked_until = NULL WHERE id = :id"),
        {"id": source.id},
    )
    db_session.commit()