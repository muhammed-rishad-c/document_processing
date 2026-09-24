import os
import secrets
import re
from typing import List
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, Header
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from .database import get_db
from .models import Company, Document, CompanyDepartment, Persona, CompanyDataSource, DocumentChunk
from .db_sync import sync_company_data_source
from .vector_store import qdrant, COLLECTION_NAME
from qdrant_client.models import Filter, FieldCondition, MatchValue
from .schemas import CompanyCreate, CompanyResponse, DepartmentsAddRequest, PersonasAddRequest
from .lead_export import LEADS_DIR

load_dotenv()  # 

INTERNAL_ADMIN_SECRET = os.getenv("INTERNAL_ADMIN_SECRET")

router = APIRouter(prefix="/internal", tags=["internal-admin"])

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
    """Manual/on-demand trigger — calls the exact same
    sync_company_data_source function the scheduled job uses, for an
    immediate refresh after a content edit without waiting for the next
    scheduled tick."""
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
        try:
            summary = sync_company_data_source(db, source)
            results.append({"source_table": source.source_table, **summary})
        except Exception as e:
            db.rollback()
            raise HTTPException(
                status_code=500,
                detail=f"Sync failed for source table '{source.source_table}': {str(e)}",
            )

    return {"company_id": company_id, "results": results}

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