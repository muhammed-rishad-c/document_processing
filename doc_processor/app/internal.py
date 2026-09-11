import os
import secrets
import re
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, Header
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from .database import get_db
from .models import Company, Document
from .schemas import CompanyCreate, CompanyResponse
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
        api_key=secrets.token_urlsafe(32),
        document_id=payload.document_id,
        allowed_origins=payload.allowed_origins,
        is_active=True,
    )
    db.add(company)
    db.commit()
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