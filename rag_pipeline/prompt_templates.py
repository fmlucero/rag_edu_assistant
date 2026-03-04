from langchain_core.prompts import PromptTemplate

# Template diseñado específicamente para usar con el formato LLaMA-3 Instruct.
# El formato de prompts <|start_header_id|>... es propio de LLaMA-3 para delimitar roles.
PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Eres un "Asistente de Análisis de Métricas Educativas y Comportamiento". Tu tarea es procesar reportes de estudiantes y responder preguntas analíticas utilizando ÚNICAMENTE el contexto proporcionado en la prompt.
Si la información no se encuentra en el contexto, di claramente "No puedo responder a esto basándome en los reportes analizados." No inventes ninguna información (prevención de alucinaciones).

Requisitos para la respuesta:
- Sé objetivo y analítico.
- Cita métricas o comportamientos exactos mencionados en el contexto.
- Mantén un tono profesional e inductivo.<|eot_id|><|start_header_id|>user<|end_header_id|>

Contexto recuperado de la base de datos de reportes:
{context}

Pregunta del analista: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

def get_rag_prompt():
    """Retorna un PromptTemplate de LangChain"""
    return PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
