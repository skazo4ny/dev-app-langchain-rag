import os
import logging

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from basic_chain import get_model
from filter import ensemble_retriever_from_docs
from local_loader import load_data_files
from memory import create_memory_chain
from rag_chain import make_rag_chain

# Configure logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_full_chain(retriever, openai_api_key, chat_memory=None):
    """Creates the full RAG chain with memory."""
    try:
        model = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=openai_api_key)
        system_prompt = """You are a helpful and knowledgeable financial consultant. 
        Use the provided context from Equity Bank's products and services to answer the user's questions. 
        If you cannot find an answer in the context, inform the user that you need more information or that the question is outside your expertise. 

        Context: {context}

        Question: {question}"""

        prompt = ChatPromptTemplate.from_template(system_prompt)

        rag_chain = make_rag_chain(model, retriever, rag_prompt=prompt)
        if chat_memory:
            return create_memory_chain(model, rag_chain, chat_memory)
        else:
            return rag_chain
    except Exception as e:
        logging.error(f"Error creating full chain: {e}")
        raise


def ask_question(chain, query, session_id):
    """Asks a question using the provided chain and session ID."""
    try:
        if chain is None:
            logging.error("Chain is None in ask_question function")
            raise ValueError("Chain is not initialized")
        logging.info(f"Invoking chain with query: {query}")
        response = chain.invoke(
            query,
            config={"configurable": {"session_id": session_id}}
        )
        logging.info("Chain invocation successful")
        return response
    except Exception as e:
        logging.error(f"Error asking question: {e}", exc_info=True)
        return {"content": "Sorry, there was an error processing your request."}


def main():
    """Main function for testing the full RAG chain."""
    load_dotenv()

    from rich.console import Console
    from rich.markdown import Markdown
    console = Console()

    try:
        docs = load_data_files()
        ensemble_retriever = ensemble_retriever_from_docs(docs)
        chain = create_full_chain(ensemble_retriever)

        queries = [ 
            "What are the benefits of opening an Equity Ordinary Account?",
            "What are the interest rates for a home loan at Equity Bank?",
            "Can you compare the Equity Gold Credit Card to the Classic Credit Card?",
            "How much does it cost to send money to an M-Pesa account using Equity Mobile Banking?",
        ]

        for query in queries:
            response = ask_question(chain, query, session_id="test_session")  # Pass a session ID
            console.print(Markdown(response.content))

    except Exception as e:
        logging.error(f"Error in main function: {e}")

if __name__ == '__main__':
    # this is to quiet parallel tokenizers warning.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()