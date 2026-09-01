import os
from uuid import UUID
from fastapi import FastApi,Depends,UploadFile,File,HTTPException,status
from sqlalchemy.orm import Session

from .database import engine,Base,get_db
from .models import Document
from .schemas import (
    DocumentResponse,
    DocumentDetailResponse,
    SearchResponse,
    SimilarityRequest,
    SimilarityResponse
)
from .service import (
    extract_text_from_file,
    calculate_document_stats,
    search_text_in_document,
    compute_tf_idf_similarity
)

Base.metadata.create_all(bind=engine)


app = FastAPI(
    title="Mini Document Processing System",
    description="API for uploading, analyzing, searching, and comparing documents.",
    version="1.0.0"
)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/documents/upload",response_model=DocumentResponse,status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile=File(...),db:Session=Depends(get_db)):
    if not file.filename:
        raise HTTPException(status_code=400,detail="Filename cannod be empty")
    
    file_bytes=await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400,detail="Uploaded file is empty")
    
    try:
        text,file_type=extract_text_from_file(file_bytes,file.filename)
        stats=calculate_document_stats(text)
    except ValueError as e:
        raise HTTPException(status_code=400,detail=str(e))
    
    doc=Document(
        filename=file.filename,
        file_type=file_type,
        extracted_text=text,
        stats=stats
    )
    
    db.add(doc)
    db.commit()
    db.refresh(doc)
    
    saved_path=os.path.join(UPLOAD_DIR,f"{doc.id}_{file.filename}")
    with open(saved_path,"wb") as f:
        f.write(file_bytes)
        
    return doc

@app.get("/documents",response_model=list[DocumentResponse])
def list_documents(dp:Session = Depends(get_db)):
    return dp.query(Document).all()
    
@app.get("/documents/{doc_id}", response_model=DocumentDetailResponse)
def get_document(doc_id: UUID, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    return doc

@app.delete("/documents/{doc_id}", status_code=status.HTTP_200_OK)
def delete_document(doc_id: UUID, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    saved_path = os.path.join(UPLOAD_DIR, f"{doc.id}_{doc.filename}")
    if os.path.exists(saved_path):
        os.remove(saved_path)

    db.delete(doc)
    db.commit()
    return {"message": "Document deleted successfully."}

@app.get("/documents/{doc_id}/search", response_model=SearchResponse)
def search_document(doc_id: UUID, query: str, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    try:
        res = search_text_in_document(doc.extracted_text, query)
        return res
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/documents/similarity", response_model=SimilarityResponse)
def document_similarity(payload: SimilarityRequest, db: Session = Depends(get_db)):
    doc1 = db.query(Document).filter(Document.id == payload.doc_id_1).first()
    doc2 = db.query(Document).filter(Document.id == payload.doc_id_2).first()

    if not doc1 or not doc2:
        raise HTTPException(status_code=404, detail="One or both documents were not found.")

    try:
        score = compute_tf_idf_similarity(doc1.extracted_text, doc2.extracted_text)
        return {
            "doc_id_1": payload.doc_id_1,
            "doc_id_2": payload.doc_id_2,
            "similarity_score": score
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    