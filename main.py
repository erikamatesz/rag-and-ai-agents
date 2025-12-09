import os
import pickle

import faiss

from rag_core import rag_retrieve
from agents import agente_prepara_aula, agente_tarefas_casa

INDEX_DIR = "index"
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.pkl")
METADATA_PATH = os.path.join(INDEX_DIR, "chunks_meta.pkl")
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")

# ======= Cores para logs (ANSI) =======
COLOR_RESET = "\033[0m"
COLOR_BLUE = "\033[94m"
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_MAGENTA = "\033[95m"
COLOR_RED = "\033[91m"
COLOR_CYAN = "\033[96m"


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
    print(f"{COLOR_BLUE}Carregando índice existente em ./index ...{COLOR_RESET}")
    index, chunks, metadata = load_index_and_data()
    if index is None or chunks is None or metadata is None:
        print(f"{COLOR_RED}Índice ou arquivos de chunks/metadados não encontrados.{COLOR_RESET}")
        print(f"{COLOR_YELLOW}Rode primeiro:{COLOR_RESET}  python index_builder.py")
        return

    print(f"{COLOR_GREEN}Índice carregado. {len(chunks)} trechos disponíveis.{COLOR_RESET}\n")

    # Mostrar quais documentos estão presentes no índice
    doc_names_unicos = sorted({m.get("doc_name") for m in metadata})
    print(f"{COLOR_CYAN}Documentos presentes no índice ({len(doc_names_unicos)}):{COLOR_RESET}")
    for name in doc_names_unicos:
        print(f" - {name}")
    print()

    print(f"{COLOR_BLUE}=== RAG + Agentes: preparação de aula e tarefas de casa ==={COLOR_RESET}")
    print(f"{COLOR_YELLOW}Digite o tema/necessidade da aula (Enter vazio para sair).{COLOR_RESET}")
    print('Exemplo: "Aula de 2h sobre mecanismos de atenção em IA para graduação"')
    print('Exemplo: "Arquitetura RAG aplicada à educação, nível graduação"\n')

    while True:
        try:
            tema = input(f"{COLOR_MAGENTA}>> Tema da aula: {COLOR_RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{COLOR_RED}Entrada interrompida. Encerrando.{COLOR_RESET}")
            break

        if not tema:
            print(f"{COLOR_GREEN}Encerrando. Até mais!{COLOR_RESET}")
            break

        print(f"\n{COLOR_BLUE}[1/3] Recuperando contextos relevantes no acervo (RAG)...{COLOR_RESET}")
        retrieved_items = rag_retrieve(tema, index, chunks, metadata)

        if not retrieved_items:
            print(f"{COLOR_RED}Nenhum contexto relevante encontrado.{COLOR_RESET}")
            continue

        # Log bonitinho: de qual PDF veio cada contexto
        print(f"{COLOR_CYAN}Contextos selecionados (após RAG):{COLOR_RESET}")
        for item in retrieved_items:
            doc_name = item.get("doc_name")
            chunk_id = item.get("chunk_id")
            rank = item.get("rank")
            print(f" - Rank {rank:02d} | doc={doc_name} | chunk={chunk_id}")
        print()

        # Texto e nomes de documentos vão para os agentes
        contextos_texto = [item["text"] for item in retrieved_items]
        nomes_docs = [item.get("doc_name", "Documento_desconhecido.pdf") for item in retrieved_items]

        print(f"{COLOR_BLUE}[2/3] Gerando plano de aula (Agente de Aula)...{COLOR_RESET}")
        plano_aula = agente_prepara_aula(tema, contextos_texto, nomes_docs=nomes_docs)

        print(f"{COLOR_BLUE}[3/3] Gerando tarefas de casa (Agente de Tarefas)...{COLOR_RESET}")
        tarefas_casa = agente_tarefas_casa(tema, contextos_texto, nomes_docs=nomes_docs)

        # Exibição no terminal (mesmo já salvando em arquivos .md pelos agentes)
        print("\n" + "=" * 80)
        print(f"{COLOR_GREEN}PLANO DE AULA{COLOR_RESET}")
        print("=" * 80)
        print(plano_aula)

        print("\n" + "=" * 80)
        print(f"{COLOR_GREEN}TAREFAS DE CASA{COLOR_RESET}")
        print("=" * 80)
        print(tarefas_casa)
        print("\n")


if __name__ == "__main__":
    main()
