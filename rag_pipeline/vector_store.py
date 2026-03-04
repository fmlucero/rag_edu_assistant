from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL_ID

def create_vector_store(chunks):
    """
    Genera embeddings para los chunks y crea el índice vectorial con FAISS.
    """
    if not chunks:
        print("No hay chunks para indexar.")
        return None
        
    print(f"Inicializando modelo de embeddings: {EMBEDDING_MODEL_ID}...")
    # Usamos sentence-transformers directamente integrado en LangChain via HF
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_ID)
    
    print("Creando índice vectorial con FAISS (esto puede requerir algo de cómputo)...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    return vector_store

def get_retriever(vector_store, k=3):
    """
    Retorna un retriever configurado para buscar los 'k' chunks más similares (similarity search).
    """
    return vector_store.as_retriever(search_kwargs={"k": k})
