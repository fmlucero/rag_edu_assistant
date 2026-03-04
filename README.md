# Asistente RAG de Análisis Educativo 🏫🤖

¡Hola! Este repo contiene un *Proof of Concept* (PoC) que estuve armando. Es básicamente un Asistente **RAG** (Retrieval-Augmented Generation) enfocado en procesar reportes o historiales de estudiantes y responder preguntas analíticas usando los datos crudos de esos textos.

La idea central acá es evitar las "alucinaciones" típicas de los LLMs. Si el profe o el analista pregunta algo sobre un alumno, el asistente busca *primero* en los reportes (usando una base de datos vectorial) y *solo* responde en base a lo que encontró ahí.

## Tecnologías que usé 🛠️
Traté de mantenerlo lo más Open Source y local posible conectando el ecosistema de Hugging Face con LangChain:

- **LangChain (LCEL)**: Orquesta todo el pipeline RAG (carga, chunking, retrieval y chains).
- **Hugging Face (`transformers` / `huggingface_hub`)**: Para cargar el LLM de forma local.
- **LLaMA-3 (Meta)**: Configurado por default para usar el modelo de Meta `Meta-Llama-3-8B-Instruct`.
- **BitsAndBytes**: Clave para la cuantización. Lo tengo configurado en **4-bit** (`nf4`) para que no te explote la PC o el server si intentás correr un modelo de 8B parámetros en hardware modesto.
- **FAISS**: Base de datos vectorial en memoria (súper rápida) para hacer el *similarity search*.
- **Sentence-Transformers**: (`all-MiniLM-L6-v2`) Un modelo rápido y liviano para convertir los textos en embeddings vectoriales y meterlos en FAISS.

## Estructura del Proyecto 📂

- `main.py`: El script principal. Corre la extracción, levanta el RAG y abre un shell interactivo en la consola para preguntar cosas.
- `config.py`: Donde seteas tokens y elegís el modelo.
- `rag_pipeline/`: El corazón del sistema.
  - `document_processor.py`: Carga recursiva y *chunking* inteligente de los textos respetando los párrafos.
  - `vector_store.py`: Inicializa la db vectorial.
  - `llm_engine.py`: Donde pasa la magia de carga del LLM con cuantización extrema.
  - `prompt_templates.py`: Prompts inyectados con el formato de LLaMA (`<|start_header_id|>`) para clavar los roles.

## Cómo correr esto 🚀

1. Cloná este repo.
2. Instalá las dependencias con (idealmente, hacete un entorno virtual):
   ```bash
   pip install -r requirements.txt
   ```
3. Agregá tus propios archivos `.txt` en la carpeta `/data`. ¡El formato da igual, pero metele reportes de ejemplo! De todos modos, el script te genera uno de muestra la primera vez.
4. Exportá tu token de Hugging Face (OBLIGATORIO si vas a usar LLaMA-3 porque es un modelo *gated* que requiere aceptar sus licencias previas en la web):
   - En Linux/Mac: `export HF_TOKEN="tu_token_aqui"`
   - En Windows (Powershell): `$env:HF_TOKEN="tu_token_aqui"`
5. Ejecutá el pipeline:
   ```bash
   python main.py
   ```

## Notas / A tener en cuenta 🤔
- **Memoria RAM/VRAM**: Aunque esté en 4-bit, un modelo de 8B requiere algo de músculo (unos ~6GB de VRAM mínimo). Si te tira errores de Out-of-Memory (OOM), fíjate en `llm_engine.py` que dejé flaggeado `llm_int8_enable_fp32_cpu_offload=True`.
- Si **no tenés HF Token** a mano o simplemente no lográs que autorice a bajar LLaMA, podés testear todo el sistema comentando el modelo de Meta en `config.py` y descomentando un modelo abierto más chico (como `Qwen/Qwen2.5-0.5B-Instruct`).
- Este PoC levanta FAISS de cero cada vez que corrés el `main.py`. Si vas a pasar esto a producción, lo ideal sería guardar el índice en disco (en `/vector_db`) o migrar a algo como ChromaDB o Milvus para no andar generando los embeddings todos los días.

¡Ojalá sirva de buena base! Pull requests y mejoras (como enchufarle Streamlit/Gradio) son más que bienvenidos. ✌️
