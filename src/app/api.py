from pathlib import Path


from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse
from src.app.core.services.indexing_service import index_pdf_file

app = FastAPI(
    title= "PDF Indexing API",
    description= "API for indexing PDF files and retrieving indexed data.",
    version= "1.0.0"
)

