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

from .models import DocumentChunk, CompanyDataSource, Company
from .service import count_token
from .vector_store import store_chunk_vector, qdrant, COLLECTION_NAME
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Same namespace convention used elsewhere for deterministic UUIDs from strings.
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


def _chunk_uuid_for_row(company_id, row_pk) -> uuid.UUID:
    return uuid.uuid5(_CHUNK_NAMESPACE, f"product:{company_id}:{row_pk}")


def sync_company_data_source(db_session, source: CompanyDataSource) -> dict:
    """
    1. SELECT * FROM {source.source_table} WHERE updated_at > last_synced_at
    2. For each changed row: format text, compute a deterministic chunk_uuid,
       upsert the DocumentChunk parent+child pair, upsert the matching Qdrant
       point (same point_id as the child chunk_uuid -> true upsert).
    3. Diff current source-table PKs against previously-synced
       source_product_id values for this company on this table; delete
       DocumentChunk rows (and their Qdrant points) for PKs no longer
       present -> handles deletes.
    4. Update source.last_synced_at = now(). Commit.

    `source.source_table` is only ever read from the CompanyDataSource row
    (never from request input), so this is safe to interpolate into the
    query — see the safety note on CompanyDataSource in models.py.
    """
    table = source.source_table
    since = source.last_synced_at or EPOCH

    # Phase 1 ships against today's single-document schema: product chunks
    # are bucketed under the company's existing Document (see the plan's
    # Phase 2 sequencing note). Phase 2 later gives the product catalog its
    # own real Document row per company — a follow-up, not a Phase 1 change.
    company = db_session.query(Company).filter(Company.id == source.company_id).first()
    if company is None or company.document_id is None:
        raise ValueError(
            f"CompanyDataSource {source.id}: company {source.company_id} has no "
            "document_id to bucket synced chunks under. A company must have a "
            "Document (via the normal upload flow) before it can sync a data source."
        )
    document_id = company.document_id

    changed_rows = db_session.execute(
        text(f'SELECT * FROM "{table}" WHERE updated_at > :since'),  # nosec: table name is trusted (see models.py)
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

        chunk_uuid = _chunk_uuid_for_row(source.company_id, row_pk)
        chunk_text = _format_row_chunk_text(row)
        token_count = count_token(chunk_text)
        content_type = row.get("content_type")

        # Delete-then-recreate the parent+child pair for this row's
        # deterministic id, so re-sync of a changed row is a clean
        # replace rather than an accumulation of stale rows.
        db_session.query(DocumentChunk).filter(
            DocumentChunk.company_id == source.company_id,
            DocumentChunk.source_product_id == row_pk,
        ).delete(synchronize_session=False)

        parent_chunk = DocumentChunk(
            id=uuid.uuid4(),
            document_id=document_id,
            chunk_index=chunk_uuid.int % (2**31),
            chunk_text=chunk_text,
            token_count=token_count,
            is_parent=True,
            parent_index=None,
            source_product_id=row_pk,
            company_id=source.company_id,
            content_type=content_type,
        )
        db_chunks_to_add.append(parent_chunk)

        child_chunk = DocumentChunk(
            id=chunk_uuid,
            document_id=document_id,
            chunk_index=chunk_uuid.int % (2**31),
            chunk_text=chunk_text,
            token_count=token_count,
            is_parent=False,
            parent_index=parent_chunk.chunk_index,
            source_product_id=row_pk,
            company_id=source.company_id,
            content_type=content_type,
        )
        db_chunks_to_add.append(child_chunk)

        vector_data.append({
            "point_id": chunk_uuid,
            "document_id": document_id,
            "company_id": source.company_id,
            "chunk_index": child_chunk.chunk_index,
            "chunk_text": chunk_text,
            "token_count": token_count,
            "parent_index": parent_chunk.chunk_index,
            "is_parent": False,
            "embedding": None,  # filled in below via batch embedding
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
