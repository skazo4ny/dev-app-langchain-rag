import os
import streamlit as st
import logging
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.embeddings import OpenAIEmbeddings
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
from openai import OpenAI
# from langchain.llms import OpenAI
from langchain_community.llms import HuggingFaceHub
from langsmith import Client

from ensemble import ensemble_retriever_from_docs
from full_chain import create_full_chain, ask_question
from local_loader import load_data_files
from memory import clean_session_history

# Read API keys at the beginning
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

st.set_page_config(page_title="Rx Example RAG")
st.title("Rx Example RAG")

def show_ui(qa, prompt_to_user="How may I help you?"):
    logging.info(f"show_ui running: {prompt_to_user}")
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": prompt_to_user}]

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
                    if qa is None:
                        raise ValueError("Chain is not initialized")
                    session_id = get_script_run_ctx().session_id
                    response = ask_question(qa, prompt, session_id)
                    if isinstance(response, dict) and 'content' in response:
                        st.markdown(response['content'])
                        message = {"role": "assistant", "content": response['content']}
                    else:
                        raise ValueError("Unexpected response format")
                except Exception as e:
                    logging.error(f"Error during question answering: {e}")
                    message = {"role": "assistant", "content": "Sorry, there was an error processing your request."}
                    st.write(message["content"])
        st.session_state.messages.append(message)

@st.cache_resource
def get_retriever():
    try:
        docs = load_data_files(data_dir="data")  
        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'), model="text-embedding-3-small")
        return ensemble_retriever_from_docs(docs, embeddings=embeddings)
    except Exception as e:
        logging.error(f"Error creating retriever: {e}")
        st.error("Error initializing the application. Please check the logs.")
        st.stop()

def check_openai_api():
    try:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Or any other suitable model
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=5
        )
        return True
    except Exception as e:
        logging.error(f"Error checking OpenAI API: {e}")
        return False

def check_huggingface_api():
    try:
        from huggingface_hub import hf_hub_download
        # Try to download a small file from a public repo
        hf_hub_download(repo_id="google/flan-t5-small", filename="config.json")
        return True
    except Exception as e:
        logging.error(f"Error checking Hugging Face API: {e}")
        return False

def check_langsmith_api():
    try:
        client = Client()
        client.list_projects()
        return True
    except Exception as e:
        logging.error(f"Error checking LangSmith API: {e}")
        return False

def get_chain():
    try:
        logging.info('Start creating chain')
        ensemble_retriever = get_retriever()
        chain = create_full_chain(
            ensemble_retriever,
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            chat_memory=StreamlitChatMessageHistory(key="langchain_messages")
        )
        logging.info('Chain creation complete')
        return ensemble_retriever, chain
    except Exception as e:
        logging.error(f"Error creating chain: {e}")
        # Print the traceback for more detailed debugging
        import traceback
        traceback.print_exc()
        st.error("Error initializing the application. Please check the logs.")
        st.stop()

def reset(prompt_to_user="How may I help you?"):
    session_id = get_script_run_ctx().session_id
    clean_session_history(session_id)
    st.session_state.messages = [{"role": "assistant", "content": prompt_to_user}]
    st.session_state['init'] = False

def run():
    logging.info("Starting run() function")
    
    if 'chain' not in st.session_state or not st.session_state.get('init', False):
        try:
            st.session_state['ensemble_retriever'], st.session_state['chain'] = get_chain()
            st.session_state['init'] = True
            logging.info("Chain initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing chain: {e}")
            st.error(f"Error initializing the application: {e}. Please check the logs.")
            return

    if not check_openai_api():
        st.error("OpenAI API key is invalid or service is unavailable.")
        st.stop()

    if not check_huggingface_api():
        st.error("Hugging Face Hub API key is invalid or service is unavailable.")
        st.stop()

    if not check_langsmith_api():
        st.error("LangSmith API key is invalid or service is unavailable.")
        st.stop()

    st.subheader("Ask questions about Equity Bank's products and services:")
    show_ui(st.session_state['chain'], "How can I assist you today?")
    st.button("Reset history", on_click=reset)
    
    logging.info("Finished run() function")

if st.button("Clear Cache and Reinitialize"):
    st.cache_resource.clear()
    st.session_state.clear()
    st.session_state['rerun'] = True

if st.session_state.get('rerun', False):
    st.session_state['rerun'] = False
    st.experimental_rerun()

run()