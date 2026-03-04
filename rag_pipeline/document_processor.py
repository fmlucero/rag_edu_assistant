import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP, DATA_DIR

def load_and_chunk_documents(data_dir=DATA_DIR):
    """
    Carga documentos de texto desde el directorio y los divide en chunks.
    Maneja reportes de estudiantes para el asistente educativo.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Directorio creado en {data_dir}. Añade archivos .txt aquí.")
        return []

    print(f"Cargando documentos desde {data_dir}...")
    loader = DirectoryLoader(data_dir, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    
    if not documents:
        print("No se encontraron documentos.")
        return []

    print(f"Se cargaron {len(documents)} documentos. Aplicando chunking...")
    
    # RecursiveCharacterTextSplitter asegura cortes más "inteligentes"
    # Respetando párrafos, frases y palabras en ese orden
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Documentos divididos en {len(chunks)} chunks.")
    return chunks
