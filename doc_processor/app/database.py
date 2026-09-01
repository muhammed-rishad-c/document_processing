import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base,sessionmaker 

database_url=os.getenv("DATABASE_URL","postgresql://postgres:rishadkhalid@localhost:5432/doc_processing_db")

engine=create_engine(database_url)
Sessionlocal=sessionmaker(autocommit=False,autoflush=False,bind=engine)
Base=declarative_base()

def get_db():
    db=Sessionlocal()
    try:
        yield db
    
    finally:
        db.close()
        
        