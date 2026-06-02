import os
import json
from dotenv import load_dotenv
from typing import List
from langchain.prompts import PromptTemplate
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_groq import ChatGroq

# Load environment
load_dotenv()

# === Setup ===
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise EnvironmentError("❌ GROQ_API_KEY not found in environment variables.")

# === Utility Functions ===

def load_text_file(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def create_vectorstore_from_text(text: str) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = [Document(page_content=chunk) for chunk in splitter.split_text(text)]
    return FAISS.from_documents(docs, embedding_model)

def save_vectorstore(vs: FAISS, path: str = "faiss_store", summary_path: str = None):
    vs.save_local(path)
    if summary_path:
        metadata_path = os.path.join(path, "metadata.json")
        summary_mtime = os.path.getmtime(summary_path)
        with open(metadata_path, "w") as f:
            json.dump({"summary_mtime": summary_mtime}, f)

def load_vectorstore(path: str = "faiss_store") -> FAISS:
    return FAISS.load_local(path, embedding_model, allow_dangerous_deserialization=True)

def is_summary_updated(summary_path: str, store_dir: str = "faiss_store") -> bool:
    metadata_path = os.path.join(store_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        return True
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
        prev_mtime = metadata.get("summary_mtime", 0)
    current_mtime = os.path.getmtime(summary_path)
    return current_mtime > prev_mtime

# === QA Chain with MultiQueryRetriever ===

def build_qa_chain(vectorstore: FAISS):
    llm = ChatGroq(
        temperature=0.3,
        api_key=GROQ_API_KEY,
        model_name="llama3-8b-8192"
    )

    # Multi-query retriever for better coverage
    retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        llm=llm
    )

    prompt = PromptTemplate.from_template(
        """You are a smart video analyst assistant with the ability to **narrate surveillance videos** just like a human narrator would during live footage.
You should **explain every visible detail, action, and interaction** in the scene clearly and naturally, as if the user is watching the video.
The user is asking questions based on **video summaries**, and your job is to **translate those summaries into rich, engaging descriptions**.

✅ Use the context to generate:
- Factual, direct, and concise answers when appropriate.
- Natural-sounding narration based on the scene summaries.
❌ Do not mention scene numbers or frame numbers.
🎯 Focus on what is happening, who is doing what, and any noticeable visual or behavioral patterns.

Context:
{context}

Question: {question}
Answer:"""
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

# === Main Function ===

def run_chatbot_from_file(summary_path: str, store_dir: str = "faiss_store"):
    # Debug file paths
    print(f"\n=== Debug Info ===")
    print(f"Summary path: {os.path.abspath(summary_path)}")
    print(f"Store dir: {os.path.abspath(store_dir)}")
    
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Summary file not found at {summary_path}")

    # Vector store handling
    if not os.path.exists(store_dir) or is_summary_updated(summary_path, store_dir):
        print("⚠️ Creating/updating FAISS store...")
        try:
            text = load_text_file(summary_path)
            if not text.strip():
                raise ValueError("Summary file is empty!")
                
            vs = create_vectorstore_from_text(text)
            print(f"Created vector store with {len(vs.docstore._dict)} documents")
            save_vectorstore(vs, store_dir, summary_path)
        except Exception as e:
            print(f"❌ Store creation failed: {e}")
            return
    else:
        print("✅ Loading existing FAISS index...")
        try:
            vs = load_vectorstore(store_dir)
            print(f"Loaded vector store with {len(vs.docstore._dict)} documents")
        except Exception as e:
            print(f"❌ Store loading failed: {e}")
            return

    # QA Chain setup
    try:
        print("\nInitializing QA chain...")
        qa_chain = build_qa_chain(vs)
        print("✅ QA chain ready!")
    except Exception as e:
        print(f"❌ QA chain failed: {e}")
        return

    # Chat loop
    print("\n🤖 Chat ready! Type 'exit' to quit")
    while True:
        try:
            query = input("You: ").strip()
            if query.lower() in ("exit", "quit"):
                break
                
            if not query:
                continue
                
            print("\n⌛ Processing...")
            result = qa_chain.invoke({"query": query})
            print(f"\n💬 Bot: {result['result']}\n")
            
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break
        except Exception as e:
            print(f"⚠️ Error: {e}\n")