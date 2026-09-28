"""DB-sourced ingestion sync (Plan Phase 1).

Generic across every company and every content type, because every source
table is expected to follow the hybrid shape from the plan's section 1.1:
always `title` + `body` + `content_type`, plus a free-form `extra` JSONB
column, plus `updated_at`.

Deterministic chunk ids (uuid5 from the company + source row pk, not
uuid4()) are what make re-sync an upsert instead of a duplicate: running
sync twice on an unchanged table is a no-op.
"""

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
    sync_start = datetime.now(timezone.utc)
    # Watermark = sync start (minus overlap), not end. A doc written mid-sync
    # isn't missed — the overlap is safe because sync is delete-then-insert
    # (idempotent). Plan section 3, decision 7.
    watermark = sync_start - timedelta(seconds=DB_SYNC_OVERLAP_SECONDS)

    col = get_mongo_db()[source.collection_name]

    document_id = _get_or_create_db_source_document(db_session, source.company_id)

    changed_docs = list(col.find({"updated_at": {"$gt": since}}).sort("updated_at", 1))

    rows_upserted = 0
    rows_deleted = 0
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

        row = {
            "id": row_pk,
            "title": doc.get("title", ""),
            "body": doc.get("body", ""),
            "content_type": doc.get("content_type") or "general",
            "extra": doc.get("extra") or {},
        }
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
    source.last_status = "ok"
    source.last_error = None
    db_session.add(source)
    db_session.commit()

    if vector_data:
        store_chunk_vector(vector_data)

    return {
        "rows_upserted": rows_upserted,
        "rows_deleted": rows_deleted,
        "duration_ms": None,
    }

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