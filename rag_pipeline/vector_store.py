import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL_ID, DB_DIR

def init_vector_store(chunks, db_dir=DB_DIR):
    """
    Inicializa el índice vectorial. Si ya existe en disco, lo carga.
    Si no, genera los embeddings para los chunks y lo guarda en disco para el futuro.
    """
    print(f"Inicializando modelo de embeddings: {EMBEDDING_MODEL_ID}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_ID)
    
    if os.path.exists(db_dir) and os.listdir(db_dir):
        print(f"[*] Base de datos vectorial encontrada en '{db_dir}'. Cargando índice cacheado para arrancar más rápido...")
        # allow_dangerous_deserialization es requerido por seguridad en LangChain para cargas locales de FAISS
        vector_store = FAISS.load_local(db_dir, embeddings, allow_dangerous_deserialization=True)
        return vector_store

    if not chunks:
        print("[!] No hay chunks ni base de datos previa para inicializar.")
        return None
        
    print("[*] Creando nuevo índice vectorial con FAISS (esto puede requerir algo de cómputo)...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Persistencia
    print(f"[*] Persistiendo índice en disco ('{db_dir}') para futuras ejecuciones...")
    os.makedirs(db_dir, exist_ok=True)
    vector_store.save_local(db_dir)
    
    return vector_store

def get_retriever(vector_store, k=3):
    """
    Retorna un retriever configurado para buscar los 'k' chunks más similares (similarity search).
    """
    return vector_store.as_retriever(search_kwargs={"k": k})
