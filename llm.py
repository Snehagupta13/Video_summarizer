from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

def build_main_chain(transcript_path: str = "src/outputs/suspicious_llm_summary_full_20250423_152616.txt"):
    # Load document
    loader = TextLoader(transcript_path)
    documents = loader.load()

    # Split text
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # Initialize HuggingFace embeddings (using sentence-transformers/all-MiniLM-L6-v2 by default)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},  # or 'cuda' if available
        encode_kwargs={'normalize_embeddings': True}
    )

    # Create vector store
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Storytelling prompt template
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are an engaging storytelling assistant. 
        Based on the transcript of a YouTube video, explain things in a friendly, clear, and detailed manner—
        just like a tour guide or narrator.

        Context: {context}

        Question: {question}
        Answer in a warm, informative tone:
        """
    )

    # Use ChatGroq LLM
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatGroq(model="llama3-8b-8192", temperature=0.7),
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=True
    )

    return qa_chain