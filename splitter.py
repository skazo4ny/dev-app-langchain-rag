# Split documents into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document


def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # Example: Increased chunk size
        chunk_overlap=100, # Example: Added overlap 
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n"]  # Example: Split at double newlines or single newlines
    )

    contents = docs
    if docs and isinstance(docs[0], Document):
        contents = [doc.page_content for doc in docs]

    texts = text_splitter.create_documents(contents)
    n_chunks = len(texts)
    print(f"Split into {n_chunks} chunks")
    return texts
