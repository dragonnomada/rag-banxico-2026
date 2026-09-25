from sentence_transformers import SentenceTransformer

modelo = SentenceTransformer(
    "../../modelos/embeddings/paraphrase-multilingual-MiniLM-L12-v2"
)

from pathlib import Path
import json

import numpy
import faiss

ruta_faiss = Path(
    "../../shared/output/financiera/faiss"
)

ruta_faiss_index = ruta_faiss / "index.faiss"
ruta_faiss_metadata = ruta_faiss / "metadata.json"

index = faiss.read_index(f"{ruta_faiss_index}")
metadata = json.loads(ruta_faiss_metadata.read_text(encoding="utf-8"))

consulta_real = "¿Quiénes pagaron el día de ayer?"
prompt_zero = "CONVIENTE ESTA PREGUNTA AGREGANDO LA FECHA DE HOY DESCRITA 2026-09-25: ¿Quiénes pagaron el día de ayer?"
consulta_one = "¿Quiénes pagaron el día de jueves veinticuatro de septiembre de 2026?"
prompt_two = "TRANSFORMA ESTA PREGUNTA SEGÚN LA ESTRUCTURA DE FAISS: <EJEMPLO> ¿Quiénes pagaron el día de jueves veinticuatro de septiembre de 2026?"
consulta_one = "Para la venta con folio <desconocido> y tres el nueve de marzo del dos mil veinti seis la cantidad de mil novecientos veinte pesos"

consulta = "FOLIO 1004"

query = modelo.encode([consulta], convert_to_numpy=True)

query = numpy.asarray(query, dtype="float32")

faiss.normalize_L2(query)

k = 5

scores, indices = index.search(query, k)

print(scores)
print(indices)

contexto = "\n\n".join(
    metadata[indice]["texto"]
    for indice in indices[0]
)

print(contexto) # CHUNK: 53 (94%): Para la venta con folio mil setenta y tres el nueve de marzo del dos mil veinti seis la cantidad de mil novecientos veinte pesos

# Consumo al LLM

import openai

cliente = openai.OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="local"
)

respuesta = cliente.chat.completions.create(
    model="local",
    messages=[
        {
            "role": "system",
            "content": "Eres un asistente de consulta a créditos financieros"
        },
        {
            "role": "user",
            "content": (
                "Intenta responder la pregunta del usuario usando el siguiente contexto"
                "CONTEXTO:\n\n"
                f"{contexto}"
                "\n\n"
                "PREGUNTA:\n\n"
                f"Datos del financiamiento 1004"
            )
        },
    ],
    response_format={
        "type": "json_schema",
        "schema": {
            "type": "array",
            "properties": {
                "folio": {
                    "type": "string",
                    "description": "folio del cliente"
                },
                "respuesta": {
                    "type": "string",
                    "description": "Respuesta a la solicitud"
                },
                "pagos": {
                    "type": "array",
                    "description": "Pagos esperados"
                },
            },
            "required": [
                "folio",
            ]
        }
    }
)

for choice in respuesta.choices:
    print(choice.message.content)
    print("-" * 80)