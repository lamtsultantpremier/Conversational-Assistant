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

import config

model = ChatOpenAI(model="gpt-4o", api_key=config.OPENAI_API_KEY)


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
        RunnablePassthrough()
        | {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | parser
    )


def get_chain2(prompt, retriever):
    return (
        RunnablePassthrough()
        | {"context": retriever | format_docs, "question": itemgetter("input")}
        | prompt
        | model
        | StructuredOutputParser()
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
