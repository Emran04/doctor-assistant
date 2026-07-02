from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from pinecone import ServerlessSpec
from dotenv import load_dotenv
import os
from src.helper import download_embeddings, filter_to_minimal_data, load_pdf_files, text_splitter

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)

extracted_data = load_pdf_files("data")
minimal_docs = filter_to_minimal_data(extracted_data)
text_chunks = text_splitter(minimal_docs)
embeddings = download_embeddings()

index_name = "medical-chatbot-dev"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        vector_type="dense",
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ),
        deletion_protection="disabled",
        tags={
            "environment": "development"
        }
    )

index = pc.Index(index_name)

docsearch = PineconeVectorStore.from_documents(text_chunks, embeddings, index_name=index_name)

print("Added documents to Pinecone index...")