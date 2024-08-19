import os
import json
from pathlib import Path

from pypdf import PdfReader
from langchain.docstore.document import Document
from langchain_community.document_loaders import TextLoader, CSVLoader, PyPDFLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader

def load_data_files(data_dir):
    """
    Loads all data files from the specified directory, handling .txt, .csv, .pdf, .xls, and .json files.

    Args:
        data_dir: The directory containing the data files.

    Returns:
        A list of Document objects, each representing a loaded document.
    """
    docs = []
    for filepath in Path(data_dir).glob('**/*.*'):
        try:
            print(f"Loading {filepath}")
            if filepath.suffix == '.txt':
                loader = TextLoader(str(filepath))
            elif filepath.suffix == '.csv':
                loader = CSVLoader(file_path=str(filepath))
            elif filepath.suffix == '.pdf':
                loader = PyPDFLoader(str(filepath))
            elif filepath.suffix == '.xls' or filepath.suffix == '.xlsx':
                loader = UnstructuredExcelLoader(str(filepath)) 
            elif filepath.suffix == '.json':
                with open(filepath) as f:
                    json_data = json.load(f)
                if isinstance(json_data, list):  # Handle list of dictionaries
                    for item in json_data:
                        content = "\n".join([f"{k}: {v}" for k, v in item.items()])
                        docs.append(Document(page_content=content, metadata={'source': str(filepath)}))
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
                    docs.append(Document(page_content=content, metadata={'source': str(filepath)}))
                else:
                    print(f"Unsupported JSON structure in {filepath}")
                continue # Skip loading with generic loader
            else:
                print(f"Unsupported file type: {filepath}")
                continue # Skip loading with generic loader
            docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    return docs



if __name__ == "__main__":
    # Test with files in the 'examples' directory
    docs = load_data_files("examples")
    for doc in docs:
        print(doc)