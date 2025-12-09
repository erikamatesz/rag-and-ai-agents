from typing import List, Dict, Any

import numpy as np
import faiss
import ollama

# Modelo de embedding usado tanto na indexação quanto na query
EMBED_MODEL = "nomic-embed-text:v1.5"
TOP_K_DEFAULT = 20

# Modelo de linguagem para tradução (pode ser o mesmo que você usa nos agentes)
LLM_MODEL_TRADUCAO = "gemma3:4b"


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


def traduzir_para_ingles(texto: str) -> str:
    """
    Traduz um texto (potencialmente em português) para inglês,
    mantendo termos técnicos de IA o mais fiéis possível.

    Se der algum erro, devolve o texto original.
    """
    try:
        prompt = f"""
Traduza o texto a seguir para INGLÊS, mantendo a terminologia técnica de Inteligência Artificial correta.
Não explique nada, responda SOMENTE com a tradução em inglês.

Texto:
\"\"\"{texto}\"\"\"
"""
        resp = ollama.chat(
            model=LLM_MODEL_TRADUCAO,
            messages=[
                {
                    "role": "system",
                    "content": "Você é um tradutor técnico PT->EN especializado em Inteligência Artificial."
                },
                {"role": "user", "content": prompt.strip()},
            ],
        )
        traducao = resp["message"]["content"].strip()
        if not traducao:
            return texto
        return traducao
    except Exception as e:
        print(f"[traduzir_para_ingles] ERRO ao traduzir, usando texto original. Detalhe: {e}")
        return texto


def rag_retrieve(
    query: str,
    index: faiss.IndexFlatL2,
    chunks: List[str],
    metadata: List[Dict[str, Any]],
    top_k: int = TOP_K_DEFAULT,
) -> List[Dict[str, Any]]:
    """
    Executa o RAG básico com suporte a queries em português:

    - Traduz a query para inglês (para combinar melhor com documentos em inglês)
    - Gera embedding da query traduzida
    - Busca no FAISS
    - Retorna uma lista de dicts com:
      {
        "text": <texto do chunk>,
        "doc_name": <nome do PDF>,
        "chunk_id": <índice do chunk dentro do doc>,
        "rank": <posição no ranking>,
        "distance": <distância L2 no índice>
      }
    """
    print(f"[rag_retrieve] Query original: {query!r}")
    query_en = traduzir_para_ingles(query)
    print(f"[rag_retrieve] Query usada para busca (EN): {query_en!r}")

    # 1) Embedding da query traduzida
    query_emb = embed_texts([query_en])  # (1, dim)
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

    # Log simples pra ver de onde vieram os resultados
    print("[rag_retrieve] Resultados (doc_name, chunk_id, rank):")
    for item in results:
        print(
            f"  - Rank {item['rank']}: doc={item['doc_name']} | "
            f"chunk={item['chunk_id']} | dist={item['distance']:.4f}"
        )

    return results
