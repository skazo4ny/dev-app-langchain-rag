import os
import json
from pathlib import Path

from pypdf import PdfReader
from langchain.docstore.document import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader


def list_txt_files(data_dir="./data"):
    paths = Path(data_dir).glob('**/*.txt')
    for path in paths:
        yield str(path)


def load_txt_files(data_dir="./data"):
    docs = []
    paths = list_txt_files(data_dir)
    for path in paths:
        print(f"Loading {path}")
        loader = TextLoader(path)
        docs.extend(loader.load())
    return docs


def load_csv_files(data_dir="./data"):
    docs = []
    paths = Path(data_dir).glob('**/*.csv')
    for path in paths:
        loader = CSVLoader(file_path=str(path))
        docs.extend(loader.load())
    return docs


# Use with result of file_to_summarize = st.file_uploader("Choose a file") or a string.
# or a file like object.
def get_document_text(uploaded_file, title=None):
    docs = []
    fname = uploaded_file.name
    if not title:
        title = os.path.basename(fname)
    if fname.lower().endswith('pdf'):
        pdf_reader = PdfReader(uploaded_file)
        for num, page in enumerate(pdf_reader.pages):
            page = page.extract_text()
            doc = Document(page_content=page, metadata={'title': title, 'page': (num + 1)})
            docs.append(doc)

    else:
        # assume text
        doc_text = uploaded_file.read().decode()
        docs.append(doc_text)

    return docs


if __name__ == "__main__":
    example_pdf_path = "examples/healthy_meal_10_tips.pdf"
    docs = get_document_text(open(example_pdf_path, "rb"))
    for doc in docs:
        print(doc)
    docs = get_document_text(open("examples/us_army_recipes.txt", "rb"))
    for doc in docs:
        print(doc)
    txt_docs = load_txt_files("examples")
    for doc in txt_docs:
        print(doc)
    csv_docs = load_csv_files("examples")
    for doc in csv_docs:
        print(doc)

def load_data_files(data_dir="./data"):
    docs = []
    paths = Path(data_dir).glob('**/*.*')  # Match all files
    for path in paths:
        try:
            print(f"Loading {path}")
            if path.suffix == '.txt':
                loader = TextLoader(path)
                docs.extend(loader.load())
            elif path.suffix == '.json':
                with open(path) as f:
                    json_data = json.load(f)
                if isinstance(json_data, list):  # Handle list of dictionaries
                    for item in json_data:
                        content = "\n".join([f"{k}: {v}" for k, v in item.items()])
                        docs.append(Document(page_content=content, metadata={'source': str(path)}))
                elif isinstance(json_data, dict):  # Handle nested dictionaries
                    content = ""
                    for key, value in json_data.items():
                        content += f"**{key}**\n\n"
                        if isinstance(value, list):
                            for item in value:
                                if isinstance(item, dict):
                                    content += "\n".join([f"{k}: {v}" for k, v in item.items()]) + "\n\n"
                                else:
                                    content += str(item) + "\n\n"
                        else:
                            content += str(value) + "\n\n"
                    docs.append(Document(page_content=content, metadata={'source': str(path)}))
                else:
                    print(f"Unsupported JSON structure in {path}")
            else:
                print(f"Unsupported file type: {path}")
        except Exception as e:
            print(f"Error loading {path}: {e}")
    return docs