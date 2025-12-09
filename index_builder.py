import os
import pickle
import json  # para salvar metadados legíveis
from typing import List, Tuple, Dict, Any

import numpy as np
import faiss
from pypdf import PdfReader

from rag_core import EMBED_MODEL
import ollama

# ======= Cores para logs (ANSI) =======
COLOR_RESET = "\033[0m"
COLOR_BLUE = "\033[94m"
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_MAGENTA = "\033[95m"
COLOR_RED = "\033[91m"

DOCS_DIR = "docs"
INDEX_DIR = "index"
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.pkl")
METADATA_PATH = os.path.join(INDEX_DIR, "chunks_meta.pkl")
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
CHUNKS_JSON_PATH = os.path.join(INDEX_DIR, "chunks_with_metadata.json")

# Chunk por caracteres
CHUNK_SIZE_CHARS = 1200
CHUNK_OVERLAP_CHARS = 300


# =========================
# CARREGAR PDFs
# =========================

def load_pdfs(docs_dir: str = DOCS_DIR) -> List[Tuple[str, str]]:
    print(f"{COLOR_BLUE}Carregando PDFs da pasta ./{docs_dir} ...{COLOR_RESET}")
    texts: List[Tuple[str, str]] = []

    if not os.path.isdir(docs_dir):
        print(f"{COLOR_RED}Pasta '{docs_dir}' não encontrada!{COLOR_RESET}")
        return texts

    for filename in os.listdir(docs_dir):
        if filename.lower().endswith(".pdf"):
            print(f"{COLOR_MAGENTA}Lendo PDF:{COLOR_RESET} {filename}")
            path = os.path.join(docs_dir, filename)
            reader = PdfReader(path)

            pages_text = []
            for page_idx, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                pages_text.append(page_text)

            full_text = "\n".join(pages_text).strip()
            if full_text:
                texts.append((filename, full_text))
            else:
                print(f"{COLOR_YELLOW}AVISO: PDF '{filename}' sem texto extraível.{COLOR_RESET}")

    print(f"{COLOR_GREEN}{len(texts)} documento(s) PDF carregado(s).{COLOR_RESET}")
    return texts


# =========================
# CHUNKING (por CARACTERES)
# =========================

def chunk_text(
    text: str,
    chunk_size_chars: int = CHUNK_SIZE_CHARS,
    overlap_chars: int = CHUNK_OVERLAP_CHARS,
) -> List[str]:

    chunks: List[str] = []
    n = len(text)
    start = 0

    while start < n:
        end = start + chunk_size_chars
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap_chars  # avança com sobreposição

    return chunks


def build_corpus_chunks(
    docs: List[Tuple[str, str]]
) -> Tuple[List[str], List[Dict[str, Any]]]:

    print(f"{COLOR_BLUE}Iniciando chunking por documento...{COLOR_RESET}\n")

    all_chunks: List[str] = []
    metadata: List[Dict[str, Any]] = []

    for doc_name, text in docs:
        print(f"{COLOR_MAGENTA}→ Fazendo chunking do documento:{COLOR_RESET} {doc_name}")
        chunks_doc = chunk_text(text)

        for i, chunk in enumerate(chunks_doc):
            all_chunks.append(chunk)
            metadata.append({"doc_name": doc_name, "chunk_id": i})

    print(f"\n{COLOR_GREEN}Total de {len(all_chunks)} chunks gerados para o corpus.{COLOR_RESET}\n")
    return all_chunks, metadata


# =========================
# EMBEDDINGS + FAISS
# =========================

def embed_chunks_with_logging(
    chunks: List[str],
    metadata: List[Dict[str, Any]],
    model: str = EMBED_MODEL,
) -> Tuple[np.ndarray, List[str], List[Dict[str, Any]]]:

    print(f"{COLOR_BLUE}Gerando embeddings com Ollama (pode levar algum tempo)...{COLOR_RESET}")

    vectors = []
    chunks_ok: List[str] = []
    metadata_ok: List[Dict[str, Any]] = []

    total = len(chunks)
    for i, (chunk, meta) in enumerate(zip(chunks, metadata)):
        print(
            f"{COLOR_YELLOW}Embedding chunk {meta.get('chunk_id')} "
            f"de '{meta.get('doc_name')}' ({i + 1}/{total})...{COLOR_RESET}"
        )

        try:
            res = ollama.embeddings(model=model, prompt=chunk)
            emb = res["embedding"]

            vectors.append(emb)
            chunks_ok.append(chunk)
            metadata_ok.append(meta)

        except Exception as e:
            print(
                f"{COLOR_RED}ERRO ao gerar embedding do chunk "
                f"(doc={meta.get('doc_name')}, chunk={meta.get('chunk_id')}). "
                f"Ignorando este chunk.\nDetalhe: {e}{COLOR_RESET}\n"
            )
            continue

    if not vectors:
        raise RuntimeError(
            f"{COLOR_RED}Nenhum embedding foi gerado — Ollama pode estar offline ou com falha.{COLOR_RESET}"
        )

    print(f"{COLOR_GREEN}Embeddings gerados com sucesso: {len(vectors)} chunks.{COLOR_RESET}")
    return np.array(vectors, dtype="float32"), chunks_ok, metadata_ok


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    print(f"{COLOR_BLUE}Construindo índice FAISS na memória...{COLOR_RESET}")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"{COLOR_GREEN}Índice FAISS criado com dimensão {dim}.{COLOR_RESET}")
    return index


def save_index_and_chunks(
    index: faiss.IndexFlatL2,
    chunks: List[str],
    metadata: List[Dict[str, Any]],
) -> None:

    print(f"{COLOR_BLUE}Salvando índice e metadados no disco...{COLOR_RESET}")
    os.makedirs(INDEX_DIR, exist_ok=True)

    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"{COLOR_GREEN}→ Índice salvo em: {FAISS_INDEX_PATH}{COLOR_RESET}")

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print(f"{COLOR_GREEN}→ Chunks salvos em: {CHUNKS_PATH}{COLOR_RESET}")

    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)
    print(f"{COLOR_GREEN}→ Metadados salvos em: {METADATA_PATH}{COLOR_RESET}")

    # JSON legível
    chunks_for_json = []
    for i, (chunk_text, meta) in enumerate(zip(chunks, metadata)):
        chunks_for_json.append({
            "global_chunk_idx": i,
            "doc_name": meta.get("doc_name"),
            "chunk_id": meta.get("chunk_id"),
            "text": chunk_text,
        })

    with open(CHUNKS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks_for_json, f, ensure_ascii=False, indent=2)

    print(f"{COLOR_GREEN}→ JSON legível salvo em: {CHUNKS_JSON_PATH}{COLOR_RESET}")


def main():
    print(f"{COLOR_BLUE}=== INICIANDO INDEXAÇÃO ==={COLOR_RESET}")

    docs = load_pdfs()
    if not docs:
        print(f"{COLOR_RED}Nenhum PDF encontrado. Abortando.{COLOR_RESET}")
        return

    chunks, metadata = build_corpus_chunks(docs)

    embeddings, chunks_ok, metadata_ok = embed_chunks_with_logging(
        chunks, metadata, model=EMBED_MODEL
    )

    index = build_faiss_index(embeddings)

    save_index_and_chunks(index, chunks_ok, metadata_ok)

    print(f"\n{COLOR_GREEN}Indexação concluída com sucesso!{COLOR_RESET}\n")

    print(f"{COLOR_BLUE}Exemplos de chunks indexados:{COLOR_RESET}")
    for i, (chunk, meta) in enumerate(zip(chunks_ok, metadata_ok)):
        if i >= 3:
            break
        preview = chunk[:200].replace("\n", " ")
        print(
            f"{COLOR_YELLOW}- global_chunk_idx={i}, doc={meta.get('doc_name')}, "
            f"chunk_id={meta.get('chunk_id')}{COLOR_RESET}\n  {preview!r}\n"
        )


if __name__ == "__main__":
    main()
