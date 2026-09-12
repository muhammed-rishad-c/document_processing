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
from .models import Company, Document, CompanyDepartment
from .schemas import CompanyCreate, CompanyResponse, DepartmentsAddRequest
from .lead_export import LEADS_DIR

load_dotenv()  # 

INTERNAL_ADMIN_SECRET = os.getenv("INTERNAL_ADMIN_SECRET")

router = APIRouter(prefix="/internal", tags=["internal-admin"])


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
    document = db.query(Document).filter(Document.id == payload.document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found.")

    company = Company(
        name=payload.name,
        tenant_id=secrets.token_urlsafe(32),
        document_id=payload.document_id,
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