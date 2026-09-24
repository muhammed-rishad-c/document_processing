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
from datetime import datetime, timezone

from sqlalchemy import text

from .models import DocumentChunk, CompanyDataSource, Company, Document
from .service import count_token, chunk_text_parent_child
from .vector_store import store_chunk_vector, qdrant, COLLECTION_NAME
from qdrant_client.models import Filter, FieldCondition, MatchValue

_CHUNK_NAMESPACE = uuid.NAMESPACE_DNS

EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

def _format_row_chunk_text(row: dict) -> str:
    """Generic row -> chunk text formatter. No changes needed when a new
    content_type or a new `extra` field shows up for any company — it just
    flows through."""
    lines = [row["title"], row["body"]]
    for key, value in (row.get("extra") or {}).items():
        lines.append(f"{key.replace('_', ' ').capitalize()}: {value}")
    return "\n".join(lines)

def _delete_row_chunks(db_session, company_id, row_pk) -> None:
    """Deletes this row's existing chunks from Postgres and their matching
    vectors from Qdrant, by chunk_id. Needed before rebuilding a row's
    chunks, because a row that changes length can split into a *different*
    number of parent/child pieces between syncs — old point ids from the
    previous split won't be overwritten by the new ones, so they'd be left
    behind as orphaned vectors if we didn't delete them explicitly first."""
    existing_children = (
        db_session.query(DocumentChunk)
        .filter(
            DocumentChunk.company_id == company_id,
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

        _delete_row_chunks(db_session, source.company_id, row_pk)


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

    # --- Diff against all previously-synced PKs for this company+table to handle deletes ---
    current_pks = {
        str(r["id"])
        for r in db_session.execute(text(f'SELECT id FROM "{table}"')).mappings().all()
    }
    stale_rows = (
        db_session.query(DocumentChunk)
        .filter(
            DocumentChunk.company_id == source.company_id,
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
            deleted_children = (
                db_session.query(DocumentChunk)
                .filter(
                    DocumentChunk.company_id == source.company_id,
                    DocumentChunk.source_product_id == pk,
                    DocumentChunk.is_parent == False,
                )
                .all()
            )
            for child in deleted_children:
                qdrant.delete(
                    collection_name=COLLECTION_NAME,
                    points_selector=Filter(
                        must=[FieldCondition(key="metadata.chunk_id", match=MatchValue(value=str(child.id)))]
                    ),
                )
            db_session.query(DocumentChunk).filter(
                DocumentChunk.company_id == source.company_id,
                DocumentChunk.source_product_id == pk,
            ).delete(synchronize_session=False)
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
