from pathlib import Path

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_chroma import Chroma

import config


model = ChatOpenAI(model="gpt-4o-mini", api_key=config.OPENAI_API_KEY)


def format_docs(docs: list[Document]):
    context = ""
    for idx, doc in enumerate(docs):
        context += (
            f"{idx}. {doc.metadata['source']}\n{doc.metadata['email']}"
            f"\n{doc.metadata['telephone']}\n{doc.metadata['siege']}"
            f"\n{doc.page_content.replace("\n\n", "\n")}\n\n"
        )
    return context


def get_chain(prompt, parser, retriever):
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RunnablePassthrough()
        | prompt
        | model
        | parser
    )


embeddings = OpenAIEmbeddings(api_key=config.OPENAI_API_KEY)
vectorstore = Chroma(
    collection_name="foncier_test_collection",
    embedding_function=embeddings,
    persist_directory=str(config.PERSIST_DIR),
)


def retrieval_chain(retriever, prompt_search_query, prompt_get_answer, document_prompt):
    retriever_chain = create_history_aware_retriever(
        model, retriever, prompt_search_query
    )

    document_chain = create_stuff_documents_chain(
        model, prompt_get_answer, document_prompt=document_prompt
    )
    return create_retrieval_chain(retriever_chain, document_chain)
