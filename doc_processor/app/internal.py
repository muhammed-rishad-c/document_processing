import os
import secrets
import re
from typing import List
from datetime import datetime, timezone
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, Header, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from .database import get_db, Sessionlocal
from .models import Company, Document, CompanyDepartment, Persona, CompanyDataSource, DocumentChunk
from .db_sync import (
    sync_company_data_source,
    sync_mongo_source,
    sync_data_source,
    acquire_sync_lease,
    release_sync_lease,
)
from .mongo_client import get_mongo_db, provision_company_collection, drop_company_collection
from .vector_store import qdrant, COLLECTION_NAME, delete_vectors_by_company, delete_vector
from qdrant_client.models import Filter, FieldCondition, MatchValue
from .schemas import (
    CompanyCreate,
    CompanyResponse,
    DepartmentsAddRequest,
    PersonasAddRequest,
    MongoContentCreate,
    MongoContentUpdate,
    MongoContentResponse,
)
from .lead_export import LEADS_DIR

load_dotenv()

INTERNAL_ADMIN_SECRET = os.getenv("INTERNAL_ADMIN_SECRET")
MONGO_ENABLED = os.getenv("MONGO_ENABLED", "false").lower() == "true"

router = APIRouter(prefix="/internal", tags=["internal-admin"])

def _get_active_mongo_source(db: Session, company_id: str) -> CompanyDataSource:
    source = (
        db.query(CompanyDataSource)
        .filter(
            CompanyDataSource.company_id == company_id,
            CompanyDataSource.source_type == "mongo",
            CompanyDataSource.is_active.is_(True),
        )
        .first()
    )
    if not source:
        raise HTTPException(status_code=404, detail="No active Mongo data source for this company.")
    return source

def _sync_mongo_source_background(company_id: str, source_id) -> None:
    """Runs after the response is already sent (FastAPI BackgroundTasks), so
    it opens its own DB session rather than reusing the request's. Takes the
    lease so it can't collide with the scheduled job syncing the same
    source at the same moment."""
    db = Sessionlocal()
    try:
        source = db.query(CompanyDataSource).filter(CompanyDataSource.id == source_id).first()
        if not source:
            return
        if not acquire_sync_lease(db, source):
            print(f"[mongo:sync] skipped company={company_id} — lease held by another run")
            return
        try:
            summary = sync_mongo_source(db, source)
            print(f"[mongo:sync] company={company_id} {summary}")
        except Exception as e:
            db.rollback()
            print(f"[mongo:sync] FAILED company={company_id}: {e}")
        finally:
            release_sync_lease(db, source)
    finally:
        db.close()

def backfill_company_id_on_chunks(db: Session, document_id, company_id) -> int:
    """Stamp company_id on every DocumentChunk belonging to `document_id`
    (Postgres) and push the same company_id into the matching Qdrant points'
    payload via set_payload — a payload-only update that does NOT re-embed
    or otherwise touch the vector itself.
    """
    chunks = db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).all()
    if not chunks:
        return 0

    for chunk in chunks:
        chunk.company_id = company_id
    db.add_all(chunks)
    db.commit()

    qdrant.set_payload(
        collection_name=COLLECTION_NAME,
        payload={"company_id": str(company_id)},
        points=Filter(
            must=[FieldCondition(key="metadata.document_id", match=MatchValue(value=str(document_id)))]
        ),
        key="metadata",
    )
    return len(chunks)

def verify_internal_secret(x_internal_secret: str = Header(...)):
    if not INTERNAL_ADMIN_SECRET:
        raise HTTPException(
            status_code=500,
            detail="Internal admin secret is not configured on the server.",
        )
    if x_internal_secret != INTERNAL_ADMIN_SECRET:
        raise HTTPException(status_code=401, detail="Invalid internal credentials.")

@router.post(
    "/companies",
    response_model=CompanyResponse,
    status_code=201,
    dependencies=[Depends(verify_internal_secret)],
)
def create_company(payload: CompanyCreate, db: Session = Depends(get_db)):
    if payload.document_id is not None:
        document = db.query(Document).filter(Document.id == payload.document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found.")

    company = Company(
        name=payload.name,
        tenant_id=secrets.token_urlsafe(32),
        allowed_origins=payload.allowed_origins,
        is_active=True,
    )
    db.add(company)
    db.flush()

    for dept in payload.departments:
        db.add(
            CompanyDepartment(
                company_id=company.id,
                name=dept.name,
                email=dept.email,
                is_default=dept.is_default,
                is_active=True,
            )
        )

    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=400,
            detail="Could not create company: department names must be unique, "
            "and exactly one department must be marked default.",
        )

    db.refresh(company)

    if payload.document_id is not None:
        document.company_id = company.id
        db.add(document)
        db.commit()
        backfill_company_id_on_chunks(db, document_id=payload.document_id, company_id=company.id)

    if MONGO_ENABLED:
        try:
            collection_name = provision_company_collection(company.id)
            db.add(CompanyDataSource(
                company_id=company.id,
                source_type="mongo",
                collection_name=collection_name,
                is_active=True,
            ))
            db.commit()
        except Exception as e:
            db.rollback()
            print(f"[mongo] provisioning failed for company={company.id}: {e}")

    return company

def _safe_filename(company_name: str) -> str:
    """Strips anything that isn't alnum/space/hyphen/underscore so the
    Content-Disposition filename can't be used to inject path separators
    or odd characters into the downloaded file's name."""
    cleaned = re.sub(r"[^A-Za-z0-9 _-]", "", company_name).strip()
    cleaned = cleaned.replace(" ", "_") or "company"
    return f"{cleaned}_leads.xlsx"

@router.get(
    "/companies",
    response_model=List[CompanyResponse],
    dependencies=[Depends(verify_internal_secret)],
)
def list_companies(db: Session = Depends(get_db)):
    return db.query(Company).all()

@router.delete(
    "/companies/{company_id}",
    dependencies=[Depends(verify_internal_secret)],
)
def delete_company(
    company_id: str,
    confirm: str,
    force: bool = False,
    db: Session = Depends(get_db),
):
    """Permanently deletes a company and everything derived from it.

    Order matters: sync leases first (so no sync can re-create chunks
    mid-delete), then the external stores (Qdrant, Mongo), and Postgres
    LAST. The Postgres row is what makes the company findable, so if any
    earlier step fails the company still exists and this call can simply
    be retried. Every step is idempotent.

    confirm: must exactly equal the company name (guards against a wrong id).
    force:   proceed even if the Mongo collection can't be dropped.
    """
    company = db.query(Company).filter(Company.id == company_id).first()
    if not company:
        raise HTTPException(status_code=404, detail="Company not found.")
    if confirm != company.name:
        raise HTTPException(status_code=400, detail="`confirm` must exactly match the company name.")

    sources = db.query(CompanyDataSource).filter(CompanyDataSource.company_id == company.id).all()

    # 1. Take every source's lease so no scheduled/manual sync runs mid-delete.
    leased = []

    def _release_all():
        for s in leased:
            try:
                release_sync_lease(db, s)
            except Exception as e:
                print(f"[delete_company] lease release failed: {e}")

    for s in sources:
        if not acquire_sync_lease(db, s):
            _release_all()
            raise HTTPException(status_code=409, detail="A sync is running for this company. Retry shortly.")
        leased.append(s)

    # 2. Stop serving it immediately, and capture uploaded-document ids
    #    (they can carry Qdrant points that were never stamped with company_id).
    company.is_active = False
    db.commit()
    doc_ids = [d for (d,) in db.query(Document.id).filter(Document.company_id == company.id).all()]
    company_id_str = str(company.id)
    company_name = company.name

    # 3. Qdrant
    try:
        delete_vectors_by_company(company_id_str)
        for doc_id in doc_ids:
            delete_vector(str(doc_id))
    except Exception as e:
        _release_all()
        raise HTTPException(status_code=502, detail=f"Qdrant cleanup failed (company kept, safe to retry): {e}")

    # 4. Mongo collections
    for s in sources:
        if s.source_type == "mongo" and s.collection_name:
            try:
                drop_company_collection(s.collection_name)
            except Exception as e:
                if not force:
                    _release_all()
                    raise HTTPException(
                        status_code=502,
                        detail=f"Mongo drop failed (company kept, safe to retry, or pass force=true): {e}",
                    )
                print(f"[delete_company] FORCED past Mongo failure, orphan collection: {s.collection_name}: {e}")

    # 5. Postgres, one transaction. Delete documents explicitly: Company.documents
    #    has no cascade, so db.delete(company) alone would NULL out
    #    documents.company_id and leave the documents behind.
    try:
        db.query(DocumentChunk).filter(DocumentChunk.company_id == company.id).delete(synchronize_session=False)
        if doc_ids:
            db.query(DocumentChunk).filter(DocumentChunk.document_id.in_(doc_ids)).delete(synchronize_session=False)
            db.query(Document).filter(Document.id.in_(doc_ids)).delete(synchronize_session=False)
        db.expire_all()
        db.delete(company)
        db.commit()
    except Exception as e:
        db.rollback()
        _release_all()
        raise HTTPException(status_code=500, detail=f"Postgres delete failed: {e}")

    # 6. Exported leads spreadsheet (contains personal data)
    try:
        (LEADS_DIR / f"{company_id_str}.xlsx").unlink(missing_ok=True)
    except Exception as e:
        print(f"[delete_company] could not remove leads file: {e}")

    return {"status": "deleted", "company": company_name, "documents_removed": len(doc_ids)}

@router.post(
    "/companies/{company_id}/departments",
    response_model=CompanyResponse,
    status_code=201,
    dependencies=[Depends(verify_internal_secret)],
)
def add_departments(company_id: str, payload: DepartmentsAddRequest, db: Session = Depends(get_db)):
    company = db.query(Company).filter(Company.id == company_id).first()
    if not company:
        raise HTTPException(status_code=404, detail="Company not found.")

    existing = (
        db.query(CompanyDepartment)
        .filter(CompanyDepartment.company_id == company_id, CompanyDepartment.is_active.is_(True))
        .all()
    )

    if len(existing) + len(payload.departments) > 10:
        raise HTTPException(status_code=400, detail="A company may have at most 10 active departments.")

    existing_names = {d.name.lower() for d in existing}
    new_names = {d.name.lower() for d in payload.departments}
    if existing_names & new_names:
        raise HTTPException(status_code=400, detail="One or more department names already exist for this company.")

    new_defaults = [d for d in payload.departments if d.is_default]
    has_existing_default = any(d.is_default for d in existing)

    if has_existing_default and new_defaults:
        raise HTTPException(
            status_code=400,
            detail="This company already has a default department. Changing the default isn't supported by this endpoint.",
        )
    if not has_existing_default and len(new_defaults) != 1:
        raise HTTPException(
            status_code=400,
            detail="This company has no default department yet — exactly one department in this request must have is_default=True.",
        )

    for dept in payload.departments:
        db.add(
            CompanyDepartment(
                company_id=company.id,
                name=dept.name,
                email=dept.email,
                is_default=dept.is_default,
                is_active=True,
            )
        )

    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=400,
            detail="Could not add departments: name conflict or default conflict.",
        )

    db.refresh(company)
    return company

@router.post(
    "/companies/{company_id}/personas",
    response_model=CompanyResponse,
    status_code=201,
    dependencies=[Depends(verify_internal_secret)],
)
def add_personas(company_id: str, payload: PersonasAddRequest, db: Session = Depends(get_db)):
    company = db.query(Company).filter(Company.id == company_id).first()
    if not company:
        raise HTTPException(status_code=404, detail="Company not found.")

    existing = (
        db.query(Persona)
        .filter(Persona.company_id == company_id, Persona.is_active.is_(True))
        .all()
    )

    if len(existing) + len(payload.personas) > 10:
        raise HTTPException(status_code=400, detail="A company may have at most 10 active personas.")

    existing_slugs = {p.slug for p in existing}
    new_slugs = {p.slug for p in payload.personas}
    if existing_slugs & new_slugs:
        raise HTTPException(status_code=400, detail="One or more persona slugs already exist for this company.")

    new_defaults = [p for p in payload.personas if p.is_default]
    has_existing_default = any(p.is_default for p in existing)

    if has_existing_default and new_defaults:
        raise HTTPException(
            status_code=400,
            detail="This company already has a default persona. Changing the default isn't supported by this endpoint.",
        )
    if not has_existing_default and len(new_defaults) != 1:
        raise HTTPException(
            status_code=400,
            detail="This company has no default persona yet — exactly one persona in this request must have is_default=True.",
        )

    for persona in payload.personas:
        db.add(
            Persona(
                company_id=company.id,
                slug=persona.slug,
                name=persona.name,
                description=persona.description,
                role_description=persona.role_description,
                greeting_text=persona.greeting_text,
                is_default=persona.is_default,
                is_active=True,
            )
        )

    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=400,
            detail="Could not add personas: slug conflict or default conflict.",
        )

    db.refresh(company)
    return company

@router.post(
    "/companies/{company_id}/sync-products",
    dependencies=[Depends(verify_internal_secret)],
)
def sync_products(company_id: str, db: Session = Depends(get_db)):
    """Manual/on-demand trigger — calls the same dispatcher the scheduled
    job uses (sync_data_source), so this now covers both Postgres and
    Mongo sources for the company, for an immediate refresh without
    waiting for the next scheduled tick. Takes the lease per source so it
    can't collide with a scheduled run hitting the same source."""
    company = db.query(Company).filter(Company.id == company_id).first()
    if not company:
        raise HTTPException(status_code=404, detail="Company not found.")

    sources = (
        db.query(CompanyDataSource)
        .filter(CompanyDataSource.company_id == company_id, CompanyDataSource.is_active.is_(True))
        .all()
    )
    if not sources:
        raise HTTPException(status_code=404, detail="No active data source configured for this company.")

    results = []
    for source in sources:
        if not acquire_sync_lease(db, source):
            results.append({"source_type": source.source_type, "skipped": "lease held by another run"})
            continue
        try:
            summary = sync_data_source(db, source)
            results.append({"source_type": source.source_type, **summary})
        except Exception as e:
            db.rollback()
            raise HTTPException(
                status_code=500,
                detail=f"Sync failed for source (type={source.source_type}): {str(e)}",
            )
        finally:
            release_sync_lease(db, source)

    return {"company_id": company_id, "results": results}

@router.post(
    "/companies/{company_id}/content",
    response_model=MongoContentResponse,
    status_code=201,
    dependencies=[Depends(verify_internal_secret)],
)
def create_mongo_content(
    company_id: str,
    payload: MongoContentCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    if not MONGO_ENABLED:
        raise HTTPException(status_code=503, detail="Mongo content is not enabled.")

    source = _get_active_mongo_source(db, company_id)
    col = get_mongo_db()[source.collection_name]

    result = col.insert_one({
        "title": payload.title,
        "body": payload.body,
        "content_type": payload.content_type,
        "extra": payload.extra,
        "is_deleted": False,
        "updated_at": datetime.now(timezone.utc),
    })

    background_tasks.add_task(_sync_mongo_source_background, company_id, source.id)
    return {"id": str(result.inserted_id)}

@router.put(
    "/companies/{company_id}/content/{content_id}",
    dependencies=[Depends(verify_internal_secret)],
)
def update_mongo_content(
    company_id: str,
    content_id: str,
    payload: MongoContentUpdate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    if not MONGO_ENABLED:
        raise HTTPException(status_code=503, detail="Mongo content is not enabled.")

    from bson import ObjectId
    from bson.errors import InvalidId

    source = _get_active_mongo_source(db, company_id)
    col = get_mongo_db()[source.collection_name]

    try:
        object_id = ObjectId(content_id)
    except InvalidId:
        raise HTTPException(status_code=404, detail="Content not found.")

    updates = {k: v for k, v in payload.dict(exclude_unset=True).items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update.")
    updates["updated_at"] = datetime.now(timezone.utc)

    result = col.update_one({"_id": object_id}, {"$set": updates})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Content not found.")

    background_tasks.add_task(_sync_mongo_source_background, company_id, source.id)
    return {"status": "updated"}

@router.delete(
    "/companies/{company_id}/content/{content_id}",
    dependencies=[Depends(verify_internal_secret)],
)
def delete_mongo_content(
    company_id: str,
    content_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    if not MONGO_ENABLED:
        raise HTTPException(status_code=503, detail="Mongo content is not enabled.")

    from bson import ObjectId
    from bson.errors import InvalidId

    source = _get_active_mongo_source(db, company_id)
    col = get_mongo_db()[source.collection_name]

    try:
        object_id = ObjectId(content_id)
    except InvalidId:
        raise HTTPException(status_code=404, detail="Content not found.")

    result = col.update_one(
        {"_id": object_id},
        {"$set": {"is_deleted": True, "updated_at": datetime.now(timezone.utc)}},
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Content not found.")

    background_tasks.add_task(_sync_mongo_source_background, company_id, source.id)
    return {"status": "deleted"}

@router.get(
    "/leads/{company_id}/download",
    dependencies=[Depends(verify_internal_secret)],
)
def download_leads(company_id: str, db: Session = Depends(get_db)):
    company = db.query(Company).filter(Company.id == company_id).first()
    if not company:
        raise HTTPException(status_code=404, detail="Company not found.")

    file_path = LEADS_DIR / f"{company_id}.xlsx"
    if not file_path.exists():
        return {"message": "No leads captured yet for this company."}

    return FileResponse(
        path=file_path,
        filename=_safe_filename(company.name),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )