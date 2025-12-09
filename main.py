import os
import pickle

import faiss

from rag_core import rag_retrieve
from agents import agente_prepara_aula, agente_tarefas_casa

INDEX_DIR = "index"
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.pkl")
METADATA_PATH = os.path.join(INDEX_DIR, "chunks_meta.pkl")
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")


def load_index_and_data():
    """
    Carrega o índice FAISS, os chunks e os metadados do disco.
    """
    if not (
        os.path.exists(FAISS_INDEX_PATH)
        and os.path.exists(CHUNKS_PATH)
        and os.path.exists(METADATA_PATH)
    ):
        return None, None, None

    index = faiss.read_index(FAISS_INDEX_PATH)

    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)

    return index, chunks, metadata


def main():
    print("Carregando índice existente em ./index ...")
    index, chunks, metadata = load_index_and_data()
    if index is None or chunks is None or metadata is None:
        print("Índice ou arquivos de chunks/metadados não encontrados.")
        print("Rode primeiro:  python index_builder.py")
        return

    print(f"Índice carregado. {len(chunks)} trechos disponíveis.\n")

    print("=== RAG + Agentes: preparação de aula e tarefas de casa ===")
    print("Digite o tema/necessidade da aula (Enter vazio para sair).")
    print('Exemplo: "Aula de 2h sobre ética em IA generativa para licenciandos"')
    print('Exemplo: "Arquitetura RAG aplicada à educação, nível graduação"\n')

    while True:
        tema = input(">> Tema da aula: ").strip()
        if not tema:
            print("Encerrando. Até mais!")
            break

        print("\n[1/3] Recuperando contextos relevantes no acervo (RAG)...")
        retrieved_items = rag_retrieve(tema, index, chunks, metadata)

        if not retrieved_items:
            print("Nenhum contexto relevante encontrado.")
            continue

        # Log bonitinho: de qual PDF veio cada contexto
        print("Contextos selecionados:")
        for item in retrieved_items:
            doc_name = item.get("doc_name")
            chunk_id = item.get("chunk_id")
            rank = item.get("rank")
            print(f"- Rank {rank} | doc={doc_name} | chunk={chunk_id}")

        # Só o texto vai para os agentes
        contextos_texto = [item["text"] for item in retrieved_items]

        print("\n[2/3] Gerando plano de aula (Agente de Aula)...")
        plano_aula = agente_prepara_aula(tema, contextos_texto)

        print("[3/3] Gerando tarefas de casa (Agente de Tarefas)...")
        tarefas_casa = agente_tarefas_casa(tema, contextos_texto)

        print("\n" + "=" * 80)
        print("PLANO DE AULA")
        print("=" * 80)
        print(plano_aula)

        print("\n" + "=" * 80)
        print("TAREFAS DE CASA")
        print("=" * 80)
        print(tarefas_casa)
        print("\n")


if __name__ == "__main__":
    main()
