import os
from langchain_community.vectorstores import FAISS 
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_PATH = os.path.join(BASE_DIR, "docs")


# Load environment variables (e.g., GOOGLE_API_KEY)
load_dotenv()

def load_documents():
    loader = DirectoryLoader(DOCS_PATH, glob="**/*.txt", loader_cls=TextLoader)
    return loader.load()

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

def embed_and_save(docs):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("vectorstore")
    print("Vectorstore saved to 'vectorstore/'")

if __name__ == "__main__":
    print("Loading and processing documents...")
    docs = load_documents()
    chunks = split_documents(docs)
    embed_and_save(chunks)