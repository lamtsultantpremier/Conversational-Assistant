from operator import itemgetter
from pathlib import Path

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import \
    create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.output_parsers import StructuredOutputParser
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
import config

from typing import List

model = ChatOpenAI(model="gpt-4o", api_key=config.OPENAI_API_KEY)

def extract_context_from_documents(documents: List[Document]) -> str:
    """
    Extrait proprement le contenu texte d'une liste de Documents LangChain
    en préservant le format d'affichage.
    """
    extracted_texts = []
    for i, doc in enumerate(documents, start=1):
        # Ajout d'un séparateur pour chaque document (utile pour le prompt)
        
        if "producer" not in doc.metadata.keys():
            section_header = f"\n--- Document {i} ---\n"
            content = doc.page_content.strip()
            provenance = doc.metadata["description"]
            email = doc.metadata["email"]
            telephone = doc.metadata["telephone"]
            source = doc.metadata["source"]
            siege = doc.metadata["siege"]
            extracted_texts.append(section_header + content + email + siege+ telephone+provenance+ "\n"+source)
    return "\n".join(extracted_texts)


def format_docs(docs: list[Document]):
    context = ""
    for idx, doc in enumerate(docs):
        context += (
            f"{idx}. {doc.metadata['source']}\n{doc.metadata['email']}"
            f"\n{doc.metadata['telephone']}\n{doc.metadata['siege']}"
            f"\n{doc.page_content.replace("\n\n", "\n")}\n\n"
        )
    return context


def get_chain(prompt, retriever):
    return (
        RunnablePassthrough()
        |{"context": itemgetter("input")|retriever | extract_context_from_documents, 
        "input": itemgetter("input"),
        "chat_history":itemgetter("chat_history")}
        | prompt
        | model
        | StrOutputParser()
    )


def get_chain2(prompt, retriever):
    return (
        RunnablePassthrough()
        | {"context": retriever | format_docs, "question": itemgetter("input")}
        | prompt
        | model
        | StrOutputParser()
    )


embeddings = OpenAIEmbeddings(api_key=config.OPENAI_API_KEY)
model_embedding = OpenAIEmbeddings(model = "text-embedding-3-large", api_key = config.OPENAI_API_KEY)
vectorstore= Chroma(
    collection_name = "final_chroma_db_database",
    persist_directory = config.PERSIST_DIR,
    embedding_function = model_embedding
)


def retrieval_chain(retriever, prompt_search_query, prompt_get_answer, document_prompt):
    retriever_chain = create_history_aware_retriever(
        model, retriever, prompt_search_query
    )

    document_chain = create_stuff_documents_chain(
        model, prompt_get_answer, document_prompt=document_prompt
    )
    return create_retrieval_chain(retriever_chain, document_chain)
