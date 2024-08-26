import os

from dotenv import load_dotenv
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages.base import BaseMessage

from basic_chain import basic_chain, get_model
from remote_loader import load_wiki_articles # updated import
from splitter import split_documents
from vector_store import create_vector_db
from local_loader import load_data_files  # Add this import

def find_similar(vs, query):
    docs = vs.similarity_search(query)
    return docs


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_question(input):
    if not input:
        return None
    elif isinstance(input,str):
        return input
    elif isinstance(input,dict) and 'question' in input:
        return input['question']
    elif isinstance(input,BaseMessage):
        return input.content
    else:
        raise Exception("string or dict with 'question' key expected as RAG chain input.")


def make_rag_chain(llm, retriever, rag_prompt=None):
    # Ensure llm and retriever are not None
    if llm is None:
        raise ValueError("LLM cannot be None")
    if retriever is None:
        raise ValueError("Retriever cannot be None")

    if rag_prompt is None:
        rag_prompt = ChatPromptTemplate.from_template(
            "Answer the question based only on the following context:\n"
            "{context}\n\n"
            "Question: {question}"
        )

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def main():
    load_dotenv()
    docs = load_data_files(data_dir="data")  # Load financial data
    texts = split_documents(docs)
    vs = create_vector_db(texts)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a financial consultant specializing in Equity Bank products."),
        ("user", "{input}")
    ])
    retriever = vs.as_retriever()

    output_parser = StrOutputParser()
    chain = basic_chain(model=get_model("ChatGPT"), prompt=prompt)  # Use your preferred model
    base_chain = chain | output_parser
    rag_chain = make_rag_chain(model=get_model("ChatGPT"), retriever=retriever) | output_parser  # Use your preferred model

    questions = [
        "What are the benefits of an Equity Ordinary Account?",
        "What are the interest rates for home loans?",
        "How do I apply for an Equity Gold Credit Card?"
    ]
    for q in questions:
        print("\n--- QUESTION: ", q)
        print("* BASE:\n", base_chain.invoke({"input": q}))
        print("* RAG:\n", rag_chain.invoke(q))


if __name__ == '__main__':
    # this is to quite parallel tokenizers warning.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()