
from datetime import datetime, timezone

# --- adjust if needed ---
from doc_processor.app.database import Sessionlocal# ------------------------

from doc_processor.app.models import Company, CompanyDataSource, DocumentChunk
from doc_processor.app.db_sync import sync_mongo_source
from doc_processor.app.mongo_client import get_mongo_db, provision_company_collection, collection_name_for
from doc_processor.app.vector_store import search_similar_chunks


def _print_chunks(db, company_id, label):
    chunks = (
        db.query(DocumentChunk)
        .filter(DocumentChunk.company_id == company_id)
        .order_by(DocumentChunk.is_parent.desc())
        .all()
    )
    print(f"\n--- {label}: {len(chunks)} chunk(s) in Postgres ---")
    for c in chunks:
        kind = "PARENT" if c.is_parent else "child"
        preview = c.chunk_text[:60].replace("\n", " ")
        print(f"  [{kind}] idx={c.chunk_index} tokens={c.token_count} text={preview!r}")
    return chunks


def main():
    db = Sessionlocal()
    company = None
    source = None
    collection_name = None

    try:
        # 1. Throwaway company
        company = Company(
            name="Mongo Sync Test Co",
            tenant_id="test-mongo-sync-tenant",
            allowed_origins=["http://localhost"],
        )
        db.add(company)
        db.flush()
        print(f"Created test company: {company.id}")

        # 2. Mongo data source row
        collection_name = provision_company_collection(company.id)
        source = CompanyDataSource(
            company_id=company.id,
            source_type="mongo",
            collection_name=collection_name,
            is_active=True,
        )
        db.add(source)
        db.commit()
        print(f"Provisioned collection: {collection_name}, source id: {source.id}")

        # 3. Insert a test doc: long body + nested extra
        long_body = (
            "QuantumTrack Warehouse Automation Suite streamlines pallet "
            "movement across large distribution centers. "
        ) * 200  # well over 2x parent_chunk_size (900), guarantees multiple parents  # comfortably over 1000 tokens once repeated
        col = get_mongo_db()[collection_name]
        insert_result = col.insert_one({
            "title": "QuantumTrack Warehouse Automation Suite",
            "body": long_body,
            "content_type": "product",
            "extra": {
                "price": "1200 USD",
                "sku": "QT-100",
                "features": ["rfid", "api", "realtime-tracking"],
                "specs": {"weight_kg": 42, "certifications": {"iso": "9001"}},
            },
            "is_deleted": False,
            "updated_at": datetime.now(timezone.utc),
        })
        doc_id = insert_result.inserted_id
        print(f"Inserted test Mongo doc: {doc_id}")

        # 4. Sync
        result = sync_mongo_source(db, source)
        print(f"\nsync_mongo_source() result: {result}")
        assert result["rows_upserted"] == 1, "expected exactly 1 row upserted"

        # 5. Check Postgres + Qdrant
        chunks = _print_chunks(db, company.id, "After insert")
        assert len(chunks) > 2, "expected multiple parent/child chunks for a long doc"
        parent_chunks = [c for c in chunks if c.is_parent]
        assert len(parent_chunks) >= 2, "expected the long body to split into multiple parents"
        # nested extra should have flattened into readable lines
        assert any("Sku" in c.chunk_text or "Specs" in c.chunk_text for c in parent_chunks), \
            "expected flattened extra fields to appear in chunk text"

        hits = search_similar_chunks(
            "warehouse automation pallet",
            top_k=5,
            company_id=str(company.id),
            db_session=db,
        )
        print(f"\nQdrant search returned {len(hits)} hit(s)")
        assert len(hits) > 0, "expected the synced content to be findable in Qdrant"

        # 6. Edit, re-sync, confirm no duplicates
        col.update_one(
            {"_id": doc_id},
            {"$set": {"title": "QuantumTrack Warehouse Automation Suite v2",
                      "updated_at": datetime.now(timezone.utc)}},
        )
        result2 = sync_mongo_source(db, source)
        print(f"\nsync_mongo_source() after edit: {result2}")
        chunks_after_edit = _print_chunks(db, company.id, "After edit")
        assert len(chunks_after_edit) == len(chunks), "chunk count changed on a same-content edit — possible duplication"

        # 7. Soft-delete, re-sync, confirm chunks are gone
        col.update_one(
            {"_id": doc_id},
            {"$set": {"is_deleted": True, "updated_at": datetime.now(timezone.utc)}},
        )
        result3 = sync_mongo_source(db, source)
        print(f"\nsync_mongo_source() after soft-delete: {result3}")
        chunks_after_delete = _print_chunks(db, company.id, "After soft-delete")
        assert len(chunks_after_delete) == 0, "expected all chunks removed after soft-delete"

        print("\n✅ All checks passed.")

    finally:
        # 8. Cleanup — always run, even on failure
        print("\nCleaning up...")
        if source is not None:
            db.query(DocumentChunk).filter(DocumentChunk.data_source_id == source.id).delete()
            db.delete(source)
        if company is not None:
            db.query(DocumentChunk).filter(DocumentChunk.company_id == company.id).delete()
            db.delete(company)
        db.commit()
        if collection_name:
            get_mongo_db().drop_collection(collection_name)
        db.close()
        print("Done.")


if __name__ == "__main__":
    main()
