import os
import secrets
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.orm import Session

from .database import get_db
from .models import Company, Document
from .schemas import CompanyCreate, CompanyResponse

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