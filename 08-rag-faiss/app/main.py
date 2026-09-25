from sentence_transformers import SentenceTransformer

modelo = SentenceTransformer(
    "../../modelos/embeddings/paraphrase-multilingual-MiniLM-L12-v2"
)

from pathlib import Path
import json

import numpy
import faiss

ruta_chunks = Path(
    "../../shared/output/financiera/chunks/contratos.json"
)

ruta_faiss = Path(
    "../../shared/output/financiera/faiss"
)

ruta_faiss.mkdir(parents=True, exist_ok=True)

ruta_faiss_index = ruta_faiss / "index.faiss"
ruta_faiss_metadata = ruta_faiss / "metadata.json"

chunks = json.loads(ruta_chunks.read_text())

textos = [chunk["texto"] for chunk in chunks]

embeddings = modelo.encode(
    textos,
    show_progress_bar=True,
    convert_to_numpy=True
)

embeddings = numpy.asarray(
    embeddings,
    dtype="float32"
)

faiss.normalize_L2(embeddings)

dimension = embeddings.shape[1]

print(f"Dimensión: {dimension}")

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

faiss.write_index(index, f"{ruta_faiss_index}")
ruta_faiss_metadata.write_text(
    json.dumps(
        chunks,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8"
)