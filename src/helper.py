from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings

# Extract text from PDF files
def load_pdf_files(path):
    loader = DirectoryLoader(path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents



def filter_to_minimal_data(documents: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list of Document objects that only contain the page_content and the source metadata.
    """
    minimal_data = []
    for doc in documents:
        minimal_doc = Document(
            page_content=doc.page_content,
            metadata={"source": doc.metadata.get("source", "")}
        )
        minimal_data.append(minimal_doc)
    return minimal_data


def text_splitter(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    return text_splitter.split_documents(documents)


def download_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings