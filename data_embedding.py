import os
import weaviate
from weaviate import EmbeddedOptions
from llama_index import download_loader
from llama_index.vector_stores import WeaviateVectorStore
from llama_index import VectorStoreIndex, StorageContext
from pathlib import Path
import argparse

INDEX = "RandstadDigital"


def get_pdf_files(base_path, loader):
    """
    Get paths to all PDF files in a directory and its subdirectories.

    Parameters:
    - base_path (str): The path to the starting directory.

    Returns:
    - list of str: A list of paths to all PDF files found.
    """
    pdf_paths = []

    # Check if the base path exists and is a directory
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"The specified base path does not exist: {base_path}")
    if not os.path.isdir(base_path):
        raise NotADirectoryError(
            f"The specified base_path is not a directory: {base_path}"
        )

    # Loop through all directories and files starting from the base path
    for root, dirs, files in os.walk(base_path):
        for filename in files:
            # If a file has a .pdf extension, add its path to the list
            if filename.endswith(".pdf"):
                pdf_file = loader.load_data(file=Path(root, filename))
                pdf_paths.extend(pdf_file)

    return pdf_paths


def main(data_dir, index_name=INDEX, query=None):
    PDFReader = download_loader("PDFReader")
    loader = PDFReader()

    documents = get_pdf_files(data_dir, loader)

    client = weaviate.Client(
        embedded_options=EmbeddedOptions(),
        additional_headers={"X-OpenAI-Api-Key": os.environ["OPENAI_API_KEY"]},
    )

    # construct vector store
    vector_store = WeaviateVectorStore(
        weaviate_client=client, index_name=index_name, text_key="content"
    )

    # setting up the storage for the embeddings
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # set up the index
    index = VectorStoreIndex(documents, storage_context=storage_context)
    if query:
        print("Running a test query to make sure the index is working...")
        query_engine = index.as_query_engine()
        response = query_engine.query(query)
        print(f"Response: {response}")
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and query PDF files.")

    parser.add_argument("--customer", default="RandstadDigital", help="Customer name")
    parser.add_argument("--pdf_dir", default="./data", help="Directory containing PDFs")
    parser.add_argument(
        "--query",
        default="What is CX0 customer exprience office?",
        help="Query to execute",
    )

    args = parser.parse_args()

    main(args)
