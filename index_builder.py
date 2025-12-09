import os
import pickle
from typing import List, Tuple, Dict, Any

import numpy as np
import faiss
from pypdf import PdfReader

from rag_core import EMBED_MODEL
import ollama


DOCS_DIR = "docs"
INDEX_DIR = "index"
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.pkl")
METADATA_PATH = os.path.join(INDEX_DIR, "chunks_meta.pkl")
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")

# Se continuar dando erro, pode reduzir ainda mais esse tamanho
CHUNK_SIZE_WORDS = 250
CHUNK_OVERLAP_WORDS = 50


# =========================
# CARREGAR PDFs
# =========================

def load_pdfs(docs_dir: str = DOCS_DIR) -> List[Tuple[str, str]]:
    """
    Lê todos os PDFs da pasta docs e retorna uma lista de tuplas:
    (nome_arquivo, texto_do_pdf).
    """
    texts: List[Tuple[str, str]] = []
    if not os.path.isdir(docs_dir):
        return texts

    for filename in os.listdir(docs_dir):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(docs_dir, filename)
            print(f"Lendo PDF: {filename}...")
            reader = PdfReader(path)
            pages_text = []

            for page_idx, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                pages_text.append(page_text)

            full_text = "\n".join(pages_text).strip()
            if full_text:
                texts.append((filename, full_text))

    return texts


# =========================
# CHUNKING
# =========================

def chunk_text(
    text: str,
    chunk_size_words: int = CHUNK_SIZE_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
) -> List[str]:
    """
    Divide um texto em pedaços (chunks) por quantidade de palavras,
    com sobreposição simples.
    """
    words = text.split()
    chunks: List[str] = []
    start = 0
    n = len(words)

    while start < n:
        end = start + chunk_size_words
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words).strip()
        if chunk:
            chunks.append(chunk)
        # avança com sobreposição
        start = end - overlap_words

    return chunks


def build_corpus_chunks(
    docs: List[Tuple[str, str]]
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Aplica chunking em todos os documentos e devolve:
    - lista de chunks (strings)
    - lista de metadados por chunk (dicts)
      ex.: {"doc_name": "arquivo.pdf", "chunk_id": 0}
    """
    all_chunks: List[str] = []
    metadata: List[Dict[str, Any]] = []

    for doc_name, text in docs:
        print(f"Fazendo chunking do documento: {doc_name}...")
        chunks_doc = chunk_text(text)

        for i, chunk in enumerate(chunks_doc):
            all_chunks.append(chunk)
            metadata.append(
                {
                    "doc_name": doc_name,
                    "chunk_id": i,
                }
            )

    return all_chunks, metadata


# =========================
# EMBEDDINGS + FAISS
# =========================

def embed_chunks_with_logging(
    chunks: List[str],
    metadata: List[Dict[str, Any]],
    model: str = EMBED_MODEL,
) -> Tuple[np.ndarray, List[str], List[Dict[str, Any]]]:
    """
    Gera embeddings usando o Ollama, exibindo qual documento está sendo processado
    e IGNORANDO chunks que derem erro (para não derrubar toda a indexação).

    Retorna:
    - embeddings (np.ndarray)
    - chunks_ok (lista de strings, apenas os chunks que deram certo)
    - metadata_ok (lista de metadados correspondente)
    """
    vectors = []
    chunks_ok: List[str] = []
    metadata_ok: List[Dict[str, Any]] = []

    total = len(chunks)
    for i, (chunk, meta) in enumerate(zip(chunks, metadata)):
        doc_name = meta.get("doc_name")
        chunk_id = meta.get("chunk_id")

        print(
            f"Gerando embedding do chunk {chunk_id} "
            f"do documento '{doc_name}' "
            f"({i + 1}/{total})..."
        )

        try:
            res = ollama.embeddings(model=model, prompt=chunk)
            emb = res["embedding"]
            vectors.append(emb)
            chunks_ok.append(chunk)
            metadata_ok.append(meta)

        except Exception as e:
            # Aqui tratamos o erro do Ollama e seguimos em frente
            print(
                f"ERRO ao gerar embedding deste chunk "
                f"(doc={doc_name}, chunk={chunk_id}). "
                f"Ele será IGNORADO.\nDetalhe: {e}\n"
            )
            continue

    if not vectors:
        raise RuntimeError(
            "Nenhum embedding foi gerado. "
            "Verifique se o Ollama está rodando e se o modelo de embedding está correto."
        )

    embeddings = np.array(vectors, dtype="float32")
    return embeddings, chunks_ok, metadata_ok


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """
    Cria um índice FAISS L2 simples na memória.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def save_index_and_chunks(
    index: faiss.IndexFlatL2,
    chunks: List[str],
    metadata: List[Dict[str, Any]],
) -> None:
    """
    Salva o índice FAISS, os chunks e os metadados em disco.
    """
    os.makedirs(INDEX_DIR, exist_ok=True)

    # salva o índice FAISS
    faiss.write_index(index, FAISS_INDEX_PATH)

    # salva os chunks (apenas o texto)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    # salva os metadados em arquivo separado
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print(f"Índice salvo em: {FAISS_INDEX_PATH}")
    print(f"Chunks salvos em: {CHUNKS_PATH}")
    print(f"Metadados dos chunks salvos em: {METADATA_PATH}")


def main():
    print("Carregando PDFs da pasta ./docs ...")
    docs = load_pdfs()
    if not docs:
        print("Nenhum PDF encontrado em ./docs. Adicione materiais sobre IA.")
        return

    print(f"{len(docs)} documento(s) PDF carregado(s).")
    print("Iniciando chunking por documento...\n")

    chunks, metadata = build_corpus_chunks(docs)
    print(f"\nTotal de {len(chunks)} trecho(s) gerado(s) para o corpus.\n")

    print("Gerando embeddings com Ollama (pode levar alguns instantes)...")
    embeddings, chunks_ok, metadata_ok = embed_chunks_with_logging(
        chunks, metadata, model=EMBED_MODEL
    )

    print("Construindo índice FAISS na memória...")
    index = build_faiss_index(embeddings)

    print("Salvando índice, chunks e metadados em disco...")
    save_index_and_chunks(index, chunks_ok, metadata_ok)

    print("\nIndexação concluída com sucesso!")


if __name__ == "__main__":
    main()
