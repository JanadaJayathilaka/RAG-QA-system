from pathlib import Path


from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse
from src.app.core.services.indexing_service import index_pdf_file
from src.app.models import QuestionRequest, QAResponse
from src.app.core.services.qa_service import answer_question
app = FastAPI(
    title= "PDF Indexing API",
    description= "API for indexing PDF files and retrieving indexed data.",
    version= "1.0.0"
)

@app.post("/index-pdf", status_code = status.HTTP_200_OK)
async def index_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are allowed.")
    
    try:
        # Save the uploaded file to a temporary location
        upload_dir = Path("data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)

        file_path = upload_dir /file.filename
        contents = await file.read()
        file_path.write_bytes(contents)

        #index the PDF file
        chunks_indexed  = index_pdf_file(file_path)

        
        return {
            "filename": file.filename,
            "chunks_indexed": chunks_indexed,
            "message": f"File '{file.filename}' indexed successfully with {chunks_indexed} chunks."
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/qa", status_code = status.HTTP_200_OK)
async def qa_endpoint(payload: QuestionRequest) -> QAResponse:
    """Submit a question about the vector databases paper.

    US-001 requirements:
    - Accept POST requests at `/qa` with JSON body containing a `question` field
    - Validate the request format and return 400 for invalid requests
    - Return 200 with `answer`, `draft_answer`, and `context` fields
    - Delegate to the multi-agent RAG service layer for processing
    """
    question = payload.question.strip() 
    if not question:
        raise HTTPException(status_code=400, detail="Missing 'question' in request body.")
    
    result = answer_question(question)
    
    return QAResponse(
        answer=result.get("answer", ""),
        context=result.get("context", "") 
    )
    