import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline
from config import MODEL_ID, HF_TOKEN

def initialize_llm():
    """
    Inicializa el modelo LLaMA desde Hugging Face usando cuantización en 4-bit con BitsAndBytes.
    """
    print(f"Inicializando LLM: {MODEL_ID} con cuantización 4-bit...")
    
    if not HF_TOKEN:
        print("ADVERTENCIA: No se encontró HF_TOKEN en las variables de entorno.")
        print("Si el modelo (ej. LLaMA-3) es restringido, la descarga fallará.")

    # 1. Configuración de Cuantización (Carga en 4-bit solo para LLaMA)
    if "Qwen" not in MODEL_ID:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, # Usa bfloat16 si la GPU lo soporta (ej. Serie RTX 30/40)
            llm_int8_enable_fp32_cpu_offload=True
        )
        quant_kwargs = {"quantization_config": bnb_config}
    else:
        # Los modelos pequeños como Qwen 0.5B pueden correr en precisión nativa en CPU/RAM
        quant_kwargs = {}

    # 2. Cargar el Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, 
        token=HF_TOKEN
    )
    
    # 3. Cargar el Modelo de Inferencia Cuantizado
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto", # Distribuye capas automáticamente entre GPU y RAM según el espacio
        token=HF_TOKEN,
        low_cpu_mem_usage=True,
        **quant_kwargs
    )

    # 4. Crear el pipeline de generación de texto local
    # Ajustamos temperatura baja para casos analíticos, reduciendo alucinaciones.
    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.1, 
        do_sample=True,
        top_p=0.9,
    )

    # 5. Adaptar el pipeline a la interfaz genérica de LangChain
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    return llm
