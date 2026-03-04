import sys
from rag_pipeline.document_processor import load_and_chunk_documents
from rag_pipeline.vector_store import init_vector_store, get_retriever
from rag_pipeline.llm_engine import initialize_llm
from rag_pipeline.prompt_templates import get_rag_prompt
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def create_sample_data():
    """Genera datos de muestra automáticamente para la ejecución del PoC."""
    import os
    from config import DATA_DIR
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    sample_path = os.path.join(DATA_DIR, "reporte_juan_perez.txt")
    if not os.path.exists(sample_path):
        with open(sample_path, "w", encoding="utf-8") as f:
            f.write("""Reporte Trimestral - Estudiante: Juan Pérez (ID: 1042)
Fecha: 15 de Noviembre de 2023

1. Rendimiento Académico:
El estudiante Juan Pérez ha mostrado una mejora significativa en Matemáticas, subiendo su calificación de 6.5 a 8.2 en el último trimestre. Su participación en la clase de Ciencias es sobresaliente, entregando todos los laboratorios a tiempo. Sin embargo, en Literatura presenta un retraso del 40% en las entregas de material asignado.

2. Comportamiento y Métricas Sociales:
Juan se integra bien con sus compañeros durante los trabajos en equipo, asumiendo con frecuencia un rol de liderazgo nato. El registro de asistencia muestra 2 ausencias injustificadas este mes. Se observa alta actividad e inquietud motora en las primeras horas de la mañana, lo que reduce temporalmente su umbral de concentración.

3. Análisis y Recomendaciones:
Se sugiere una tutoría de apoyo focalizada para Literatura y que el equipo docente rotativo monitoree los niveles de atención durante el primer bloque horario.
""")

def main():
    print("="*60)
    print(" INICIANDO PoC: Asistente RAG Educativo (Llama-3)".center(60))
    print("="*60)
    
    # 1. Crear documento de muestra para test
    create_sample_data()

    # 2. Carga, procesamiento y Chunking
    chunks = load_and_chunk_documents()
    if not chunks:
        print("[!] Por favor añade documentos .txt en la carpeta 'data' y vuelve a intentar.")
        sys.exit(1)

    # 3. Creación o Carga de Index Vectorial persistido
    vector_store = init_vector_store(chunks)
    retriever = get_retriever(vector_store, k=2)

    # 4. Carga y Cuantización del LLM local de Hugging Face
    try:
        llm = initialize_llm()
    except Exception as e:
        print(f"\n[ERROR CRÍTICO] Hubo un problema al cargar el LLM.")
        print("Tip 1: ¿Ejecutaste 'pip install -r requirements.txt'?")
        print("Tip 2: Necesitas 'export HF_TOKEN=tu_token' si usas LLaMA-3 cerrado.")
        print(f"Detalles: {e}")
        sys.exit(1)

    # 5. Ensamblaje de la Cadena RAG con LangChain (LCEL)
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | get_rag_prompt()
        | llm
        | StrOutputParser()
    )

    print("\n" + "="*60)
    print("✅ SISTEMA RAG LISTO Y OPERATIVO")
    print("Escribe 'salir' para terminar la ejecución.")
    print("Prueba preguntando: '¿Cuál es el promedio de matemáticas de Juan?'")
    print("="*60 + "\n")

    # 6. Modo Interactivo de Búsqueda y Generación
    while True:
        try:
            question = input("\n🧑‍🏫 Analista Educativo> ")
            if question.lower() in ['salir', 'exit', 'quit']:
                print("Finalizando sistema...")
                break
                
            if not question.strip():
                continue

            print("🧠 Procesando embeddings y generando respuesta...")
            # Invocación de base de datos vectorial + LLM
            docs = retriever.invoke(question)
            result = qa_chain.invoke(question)
            
            print("\n" + "-"*50)
            print("🤖 RESPUESTA DEL ASISTENTE LLaMA:")
            print(result)
            
            # Verificación de transparencia RAG (Sources)
            print("\n[📁 Contexto Recuperado desde VectorDB]:")
            for i, doc in enumerate(docs, 1):
                clean_context = doc.page_content.replace('\n', ' ')
                print(f"   Fragmento {i}: ...{clean_context[:120]}...")
            print("-"*50)
            
        except KeyboardInterrupt:
            print("\nOperación cancelada. Escribe 'salir' para salir.")

if __name__ == "__main__":
    main()
