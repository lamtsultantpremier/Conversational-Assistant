import os
import time
from typing import Annotated, List

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from langchain_core.output_parsers.string import StrOutputParser

import config
from src import schemas
from src.chain import get_chain, retrieval_chain, vectorstore
from src.crud import link_crud
from src.database import Base, DbSession, engine
from src.prompts import (chat_prompt, document_prompt, prompt_get_answer,
                         prompt_search_query)

app = FastAPI()

Base.metadata.create_all(bind=engine)


# -------------Gestions des PDFS------------


@app.get("/", tags=["Main route"])
async def greet():
    return {"message": "Bienvenue dans notre API"}


@app.get(
    "/files",
    tags=["Gestion des fichiers PDF"],
    summary="Obtenir la liste des fichiers pdf dans la base",
)
async def get_all_files():
    files = [f for f in os.listdir(config.PDF_DIR) if f.endswith(".pdf")]
    return {"pdf_files": files}


@app.get(
    "/files/{filename}",
    tags=["Gestion des fichiers PDF"],
    summary="Télécharger un fichier pdf à partir de son nom",
)
async def get_file(filename: str):
    file_path = os.path.join(config.PDF_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="application/pdf", filename=filename)
    return {"error": "Fichier non trouvé"}


@app.post(
    "/upload",
    tags=["Gestion des fichiers PDF"],
    summary="Ajouter un fichier pdf à la base",
)
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type == "application/pdf" and file.filename:
        file_path = os.path.join(config.PDF_DIR, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
    else:
        raise HTTPException(status_code=500, detail="Only '.pdf' files are allowed")

    return {"message": f"{file.filename} uploadé avec succès"}


# --------------Gestions des liens------------


@app.get("/links/", response_model=List[schemas.LinkRead], tags=["Gestion des liens"])
async def read_links(db: DbSession):
    links = link_crud.get_links(db)
    return links


@app.post("/links/", response_model=schemas.LinkRead, tags=["Gestion des liens"])
async def creacte_link(db: DbSession, url: Annotated[schemas.LinkCreate, Query()]):
    link = link_crud.create_link(db, url)
    return link


# --------------RAG------------


@app.post("/foncier/chats", tags=["RAG Chat with HISTORY "])
async def rag_with_history(input: schemas.QuestionInput):
    # chain = retrieval_chain(
    #     retriever=vectorstore.as_retriever(),
    #     prompt_search_query=prompt_search_query,
    #     prompt_get_answer=prompt_get_answer,
    #     document_prompt=document_prompt,
    # )
    retriever = vectorstore.as_retriever()
    chain = get_chain(prompt_get_answer , retriever)
    response = chain.invoke(
        {"chat_history": input.chat_history, "input": input.question}
    )
    return response
