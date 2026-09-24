import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client.models import Filter, FieldCondition, MatchValue

from doc_processor.app.database import Sessionlocal
from doc_processor.app.models import DocumentChunk
from doc_processor.app.db_sync import _get_or_create_db_source_document
from doc_processor.app.vector_store import qdrant, COLLECTION_NAME


def migrate_product_chunks() -> None:
    db = Sessionlocal()
    try:
        company_ids = {
            row.company_id
            for row in db.query(DocumentChunk.company_id)
            .filter(DocumentChunk.source_product_id.isnot(None))
            .distinct()
            .all()
            if row.company_id is not None
        }

        for company_id in company_ids:
            new_document_id = _get_or_create_db_source_document(db, company_id)

            chunks = (
                db.query(DocumentChunk)
                .filter(
                    DocumentChunk.company_id == company_id,
                    DocumentChunk.source_product_id.isnot(None),
                    DocumentChunk.document_id != new_document_id,
                )
                .all()
            )
            if not chunks:
                print(f"[migrate] company={company_id}: already migrated, skipping")
                continue

            old_document_ids = {str(c.document_id) for c in chunks}
            for chunk in chunks:
                chunk.document_id = new_document_id
            db.add_all(chunks)
            db.commit()

            for chunk in chunks:
                if chunk.is_parent:
                    continue
                qdrant.set_payload(
                    collection_name=COLLECTION_NAME,
                    payload={"metadata.document_id": str(new_document_id)},
                    points=Filter(
                        must=[FieldCondition(key="metadata.chunk_id", match=MatchValue(value=str(chunk.id)))]
                    ),
                )

            print(f"[migrate] company={company_id}: moved {len(chunks)} chunks "
                  f"from {old_document_ids} -> {new_document_id}")
    finally:
        db.close()


if __name__ == "__main__":
    migrate_product_chunks()