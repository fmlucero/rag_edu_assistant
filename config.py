import os

# Configuraciones del Modelo LLM
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
# MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct" # Modelo ungated usado para pruebas locales
HF_TOKEN = os.getenv("HF_TOKEN") # Asegúrate de configurar la variable de entorno HF_TOKEN

# Configuraciones de Embeddings
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# Configuraciones de Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Rutas de datos
DATA_DIR = "./data"
DB_DIR = "./vector_db"
