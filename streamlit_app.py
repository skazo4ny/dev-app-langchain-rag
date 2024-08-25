import os
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import logging
import asyncio
from chromadb.api.models.Collection import Collection
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from pathlib import Path
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
from chromadb import PersistentClient

# Langchain tracing
from langsmith.run_helpers import traceable
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

from ensemble import ensemble_retriever_from_docs
from full_chain import create_full_chain, ask_question
from local_loader import load_data_files, load_file
from vector_store import EmbeddingProxy 
from memory import clean_session_history

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

st.set_page_config(page_title="Rx Example RAG")
st.title("Rx Example RAG")

def show_ui(qa, prompt_to_user="How may I help you?"):
    """
    Displays the Streamlit chat UI and handles user interactions.

    Args:
        qa: The LangChain chain for question answering.
        prompt_to_user: The initial prompt to display to the user.
    """
    logging.info(f"show_ui running: {prompt_to_user}")
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": prompt_to_user}]

    # Add LangSmith tracing callback
    st_callback = StreamlitCallbackHandler(st.container())

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    session_id = get_script_run_ctx().session_id
                    response = ask_question(qa, prompt, session_id, callbacks=[st_callback])
                    content = response.get("content", "No response content")
                    st.markdown(content)
                    message = {"role": "assistant", "content": content}
                except Exception as e:
                    logging.error(f"Error during question answering: {e}")
                    error_message = "Sorry, there was an error processing your request."
                    st.write(error_message)
                    message = {"role": "assistant", "content": error_message}
        st.session_state.messages.append(message)

@st.cache_resource
def get_retriever(openai_api_key=None):
    """
    Creates and caches the document retriever.

    Args:
        openai_api_key: The OpenAI API key.

    Returns:
        An ensemble document retriever.
    """
    try:
        docs = load_data_files(data_dir="data")  
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-3-small") 
        return ensemble_retriever_from_docs(docs, embeddings=embeddings)
    except Exception as e:
        logging.error(f"Error creating retriever: {e}")
        logging.exception("Exception details:")
        st.error("Error initializing the application. Please check the logs.")
        st.stop() 


def get_chain(openai_api_key=None, huggingfacehub_api_token=None):
    try:
        ensemble_retriever = get_retriever(openai_api_key=openai_api_key)
        chain = create_full_chain(
            ensemble_retriever,
            openai_api_key=openai_api_key,
            chat_memory=StreamlitChatMessageHistory(key="langchain_messages")
        )
        return ensemble_retriever, chain
    except Exception as e:
        logging.error(f"Error creating chain: {e}")
        logging.exception("Exception details:")
        st.error("Error initializing the application. Please check the logs.")
        raise

def get_secret_or_input(secret_key, secret_name, info_link=None):
    """
    Retrieves a secret from Streamlit secrets or prompts the user for input.

    Args:
        secret_key: The key of the secret in Streamlit secrets.
        secret_name: The user-friendly name of the secret.
        info_link: An optional link to provide information about the secret.

    Returns:
        The secret value.
    """
    if secret_key in st.secrets:
        st.write("Found %s secret" % secret_key)
        secret_value = st.secrets[secret_key]
    else:
        st.write(f"Please provide your {secret_name}")
        secret_value = st.text_input(secret_name, key=f"input_{secret_key}", type="password")
        if secret_value:
            st.session_state[secret_key] = secret_value
        if info_link:
            st.markdown(f"[Get an {secret_name}]({info_link})")
    return secret_value

async def process_uploaded_file_async(uploaded_file, openai_api_key=None):
    """
    Processes the uploaded file and adds it to the vector database.

    Args:
        uploaded_file: The uploaded file object from Streamlit.
        openai_api_key: The OpenAI API key for embedding generation.
    """
    try:
        if uploaded_file is not None:
            # Get the file path from the uploaded file object
            file_path = os.path.join("data", uploaded_file.name) 

            # Save the uploaded file to the 'data' directory
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Load the document using the saved file path
            docs = load_data_files(data_dir=os.path.dirname(file_path))

            # Use the same embedding model as in vector_store.py
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-3-small")
            proxy_embeddings = EmbeddingProxy(embeddings)  

            # Get the persistent Chroma client 
            persist_directory = os.path.join("store", "chroma")
            client = PersistentClient(path=persist_directory)

            # Get or create the collection
            collection = client.get_or_create_collection(name="chroma", embedding_function=proxy_embeddings)

            # Generate unique IDs for the documents (using UUIDs)
            import uuid
            tasks = [asyncio.create_task(process_document(doc, proxy_embeddings)) for doc in docs]
            processed_docs = await asyncio.gather(*tasks)

            # Filter out None values and add only successfully processed documents
            valid_docs = [doc for doc in processed_docs if doc is not None]

            if valid_docs:
                collection.add(
                    ids=[doc['id'] for doc in valid_docs],
                    embeddings=[doc['embedding'] for doc in valid_docs],
                    documents=[doc['content'] for doc in valid_docs],
                    metadatas=[doc['metadata'] for doc in valid_docs]
                )
                st.success(f"File uploaded and {len(valid_docs)} documents added to the knowledge base!")
            else:
                st.warning("No documents were successfully processed and added to the knowledge base.")

    except Exception as e:
        logging.error(f"Error processing uploaded file: {e}")
        st.error("Error processing the file. Please check the logs.")

async def process_document(doc, proxy_embeddings):
    try:
        doc.metadata['id'] = str(uuid.uuid4())
        embedding = await proxy_embeddings.aembed_query(doc.page_content)
        return {
            'id': doc.metadata['id'],
            'embedding': embedding,
            'content': doc.page_content,
            'metadata': doc.metadata
        }
    except Exception as e:
        logging.error(f"Error processing document: {e}")
        return None

def reset(prompt_to_user="How may I help you?"):
    session_id = get_script_run_ctx().session_id
    clean_session_history(session_id)
    st.session_state.messages = [{"role": "assistant", "content": prompt_to_user}]
    st.session_state['init'] = False  # Force reinitialization of the chain

@traceable
def run():
    """
    Main function to run the Streamlit application.
    """
    ready = True
    openai_api_key = st.session_state.get("OPENAI_API_KEY")
    huggingfacehub_api_token = st.session_state.get("HUGGINGFACEHUB_API_TOKEN")
    langchain_api_key = st.session_state.get("LANGCHAIN_API_KEY")

    with st.sidebar:
        if not openai_api_key:
            openai_api_key = get_secret_or_input(
                'OPENAI_API_KEY',
                "OpenAI API key",
                info_link="https://platform.openai.com/account/api-keys"
            )
        if not huggingfacehub_api_token:
            huggingfacehub_api_token = get_secret_or_input(
                'HUGGINGFACEHUB_API_TOKEN',
                "HuggingFace Hub API Token",
                info_link="https://huggingface.co/docs/huggingface_hub/main/en/quick-start#authentication"
            )
        if not langchain_api_key:
            langchain_api_key = get_secret_or_input(
                'LANGCHAIN_API_KEY',
                "LangSmith API Key",
                info_link="https://docs.langchain.com/docs/tracing/getting_started#setting-up-tracing"
            )

    if not openai_api_key:
        st.warning("Missing OPENAI_API_KEY")
        ready = False
    if not huggingfacehub_api_token:
        st.warning("Missing HUGGINGFACEHUB_API_TOKEN")
        ready = False
    if not langchain_api_key:
        st.warning("Missing LANGCHAIN_API_KEY")
        ready = False

    uploaded_file = st.file_uploader("Choose a file to upload", type=["txt", "pdf"])
    if uploaded_file:
        asyncio.run(process_uploaded_file_async(uploaded_file, openai_api_key))

    if ready:
        try:
            logging.info('run loop')

            # Set LangSmith environment variables
            os.environ["LANGCHAIN_API_KEY"] = langchain_api_key 
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
            os.environ["LANGCHAIN_PROJECT"] = "rx-kenny-rag-streamlit-dev" # or your project name

            if not st.session_state.get('init', False):
                st.session_state['ensemble_retriever'], st.session_state['chain'] = get_chain(
                    openai_api_key=openai_api_key,
                    huggingfacehub_api_token=huggingfacehub_api_token
                )
                st.session_state['init'] = True

            # Chat Interface
            st.subheader("Ask questions about Equity Bank's products and services:")
            show_ui(st.session_state['chain'], "How can I assist you today?")
            st.button("Reset history", on_click=reset)

        except Exception as e:
            logging.error(f"Error initializing application: {e}")
            logging.exception("Exception details:") 
            st.error("Error initializing the application. Please check the logs.")
    else:
        st.stop()

run()