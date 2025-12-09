from typing import List, Dict, Any

import numpy as np
import faiss
import ollama

# Modelo de embedding usado tanto na indexação quanto na query
EMBED_MODEL = "nomic-embed-text"
TOP_K_DEFAULT = 4


def embed_texts(texts: List[str], model: str = EMBED_MODEL) -> np.ndarray:
    """
    Gera embeddings usando o Ollama.
    Retorna um array NumPy (n_texts, dim).
    """
    vectors = []
    for t in texts:
        res = ollama.embeddings(model=model, prompt=t)
        emb = res["embedding"]
        vectors.append(emb)

    return np.array(vectors, dtype="float32")


def rag_retrieve(
    query: str,
    index: faiss.IndexFlatL2,
    chunks: List[str],
    metadata: List[Dict[str, Any]],
    top_k: int = TOP_K_DEFAULT,
) -> List[Dict[str, Any]]:
    """
    Executa o RAG básico:
    - embed da query
    - busca no FAISS
    - retorna uma lista de dicts com:
      {
        "text": <texto do chunk>,
        "doc_name": <nome do PDF>,
        "chunk_id": <índice do chunk dentro do doc>,
        "rank": <posição no ranking>,
        "distance": <distância L2 no índice>
      }
    """
    query_emb = embed_texts([query])  # (1, dim)
    distances, indices = index.search(query_emb, top_k)

    results: List[Dict[str, Any]] = []
    idxs = indices[0]
    dists = distances[0]

    for rank, idx in enumerate(idxs):
        if 0 <= idx < len(chunks):
            meta = metadata[idx] if idx < len(metadata) else {}
            item: Dict[str, Any] = {
                "text": chunks[idx],
                "doc_name": meta.get("doc_name"),
                "chunk_id": meta.get("chunk_id"),
                "rank": rank,
                "distance": float(dists[rank]),
            }
            results.append(item)

    return results
