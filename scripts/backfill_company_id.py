
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from doc_processor.app.database import Sessionlocal
from doc_processor.app.models import Company
from doc_processor.app.internal import backfill_company_id_on_chunks


def run() -> None:
    db = Sessionlocal()
    try:
        companies = db.query(Company).all()
        total_chunks = 0
        for company in companies:
            if company.document_id is None:
                print(f"[backfill] company={company.id} ({company.name}) has no document_id, skipping")
                continue
            count = backfill_company_id_on_chunks(db, document_id=company.document_id, company_id=company.id)
            total_chunks += count
            print(f"[backfill] company={company.id} ({company.name}): stamped {count} chunks")
        print(f"[backfill] done — {total_chunks} chunks updated across {len(companies)} companies")
    finally:
        db.close()


if __name__ == "__main__":
    run()