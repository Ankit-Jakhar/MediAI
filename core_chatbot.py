from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from prompt import medical_prompt_template
from dotenv import load_dotenv
import os

load_dotenv()

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("vectorstore", embedding, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

hf_pipeline = pipeline("text-generation", model="microsoft/DialoGPT-small", device=-1)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": medical_prompt_template}
)

def ask_bot(query: str):
    return qa_chain.run(query)

print("llm are working")
