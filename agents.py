import os
import textwrap
import unicodedata
import re
from typing import List

import ollama

LLM_MODEL_DEFAULT = "gemma3:4b"

# ======= Cores para logs (ANSI) =======
COLOR_RESET = "\033[0m"
COLOR_BLUE = "\033[94m"
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_MAGENTA = "\033[95m"

OUTPUT_DIR = "output"


# ---------- Utilitário para criar nomes de arquivos ----------
def sanitize_filename(name: str) -> str:
    # Remove acentos
    nfkd = unicodedata.normalize("NFKD", name)
    name = "".join([c for c in nfkd if not unicodedata.combining(c)])
    # Remove caracteres estranhos
    name = re.sub(r"[^a-zA-Z0-9_\- ]", "", name)
    # Substitui espaços por _
    name = name.replace(" ", "_")
    return name.lower()


def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# AGENTE 1 – PREPARA AULA
# =========================

def agente_prepara_aula(
    tema: str,
    contextos: List[str],
    llm_model: str = LLM_MODEL_DEFAULT,
    nomes_docs: List[str] | None = None,
) -> str:

    print(f"\n{COLOR_BLUE}[agente_prepara_aula] Iniciando preparação da aula...{COLOR_RESET}")
    print(f"{COLOR_BLUE}[agente_prepara_aula] Tema recebido: {tema!r}{COLOR_RESET}")
    print(f"{COLOR_BLUE}[agente_prepara_aula] Quantidade de contextos recebidos: {len(contextos)}{COLOR_RESET}")
    print(f"{COLOR_BLUE}[agente_prepara_aula] Modelo LLM usado: {llm_model!r}{COLOR_RESET}")

    if nomes_docs is None:
        nomes_docs = ["Documento_desconhecido.pdf"] * len(contextos)

    # lista única de arquivos para o prompt
    arquivos_unicos = sorted(set(n for n in nomes_docs if n is not None))

    # Mostrar prévia dos contextos
    for i, (ctx, doc_name) in enumerate(zip(contextos, nomes_docs), start=1):
        preview = ctx[:200].replace("\n", " ")
        print(
            f"{COLOR_YELLOW}[agente_prepara_aula] Prévia do contexto {i} "
            f"({doc_name}): {preview!r}...{COLOR_RESET}"
        )

    # Cada contexto recebe um cabeçalho com a fonte
    # formato: [FONTE: nome_do_arquivo.pdf]
    contextos_marcados = []
    for ctx, doc_name in zip(contextos, nomes_docs):
        bloco = f"[FONTE: {doc_name}]\n{ctx}"
        contextos_marcados.append(bloco)

    context = "\n\n---\n\n".join(contextos_marcados)

    # Monta lista de arquivos para o prompt
    arquivos_str = "\n".join(f"- {nome}" for nome in arquivos_unicos)

    user_prompt = f"""
Você é um professor universitário de Inteligência Artificial.

Use EXCLUSIVAMENTE o contexto abaixo (trechos extraídos dos PDFs do professor)
para preparar um plano de aula universitária.

Cada bloco de contexto começa com uma linha no formato:
[FONTE: NOME_DO_ARQUIVO.pdf]

ARQUIVOS DISPONÍVEIS (USE APENAS ESSES NOMES, COPIANDO-OS EXATAMENTE):
{arquivos_str if arquivos_str else "- (nenhum nome de arquivo disponível)"}

REGRAS SOBRE NOMES DE ARQUIVOS:
- Você SÓ pode citar arquivos usando exatamente um dos nomes listados acima.
- NÃO invente nomes de arquivos, abreviações, rótulos genéricos ou números.
- NÃO use "Arquivo PDF 1", "Arquivo PDF 2", "Documento 1", "Documento X" etc.
- Se precisar citar um arquivo, escolha um nome **exatamente** da lista ARQUIVOS DISPONÍVEIS.
- Se a informação não puder ser associada a nenhum arquivo da lista, NÃO invente um nome — apenas explique o conteúdo sem atribuir a um arquivo específico ou diga que não há arquivo associado.

REGRAS GERAIS IMPORTANTES:
- NÃO invente conceitos, fatos históricos, autores ou algoritmos que NÃO estejam nos contextos fornecidos.
- Se alguma informação não estiver claramente presente nos materiais, diga explicitamente:
  "Esta informação não foi encontrada nos materiais fornecidos."
- Mantenha-se fiel ao conteúdo dos arquivos. Não crie conteúdo que contradiga os textos.

CONTEXTOS:
{context}

TAREFA:
Crie um plano de aula com:
- Título da aula
- Objetivos de aprendizagem (indicando de quais arquivos vieram os principais conceitos)
- Pré-requisitos
- Conteúdos principais
- Metodologia (tempo estimado)
- Atividades em sala (podendo incluir uso de IA generativa)
- Leituras obrigatórias e complementares
- Formas de avaliação

SEÇÕES ADICIONAIS OBRIGATÓRIAS:
1. "Documentos consultados": liste apenas os nomes dos arquivos PDF REALMENTE utilizados na aula.
   - Nesta lista, COPIE EXATAMENTE os nomes que aparecem em ARQUIVOS DISPONÍVEIS.
   - Não crie rótulos genéricos (nada de "Arquivo PDF 1", "Arquivo 2"...).
2. "Trechos citados dos materiais": inclua pequenas citações (1–3 linhas)
   dos textos, sempre indicando entre parênteses o nome do arquivo de origem,
   também COPIADO exatamente como aparece em ARQUIVOS DISPONÍVEIS.

FORMATO DA RESPOSTA:
- A resposta deve ser SOMENTE o plano de aula final, em Markdown.
- NÃO inclua frases como "Segue o plano de aula:", "Como solicitado", "Conclusão" fora da estrutura normal.
- NÃO explique o que você está fazendo, apenas apresente o plano de aula organizado.
- Toda a resposta deve estar em português (termos técnicos podem aparecer em inglês se assim estiverem nos arquivos).

Tema da aula:
\"\"\"{tema}\"\"\"

Escreva em português, com seções bem definidas em Markdown (## Títulos, ### subtítulos).
"""

    print(f"{COLOR_MAGENTA}[agente_prepara_aula] Chamando o modelo Ollama...{COLOR_RESET}")
    response = ollama.chat(
        model=llm_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Você é um especialista em didática universitária de Inteligência Artificial. "
                    "Você NUNCA inventa informações que não estejam nos contextos fornecidos, "
                    "NUNCA inventa nomes de arquivos e SEMPRE indica claramente o nome do arquivo PDF "
                    "de origem, copiando-o exatamente da lista fornecida. "
                    "Sua resposta deve ser apenas o plano de aula final, sem comentários adicionais."
                ),
            },
            {"role": "user", "content": textwrap.dedent(user_prompt).strip()},
        ],
    )
    conteudo_resposta = response["message"]["content"]

    print(f"{COLOR_GREEN}[agente_prepara_aula] Plano de aula gerado. Salvando arquivo MD...{COLOR_RESET}")

    ensure_output_dir()
    filename = os.path.join(OUTPUT_DIR, f"plano_de_aula_{sanitize_filename(tema)}.md")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(conteudo_resposta)

    print(f"{COLOR_GREEN}[agente_prepara_aula] Arquivo salvo: {filename}{COLOR_RESET}")

    return conteudo_resposta


# =========================
# AGENTE 2 – TAREFAS DE CASA
# =========================

def agente_tarefas_casa(
    tema: str,
    contextos: List[str],
    llm_model: str = LLM_MODEL_DEFAULT,
    nomes_docs: List[str] | None = None,
) -> str:

    print(f"\n{COLOR_BLUE}[agente_tarefas_casa] Iniciando criação das tarefas de casa...{COLOR_RESET}")
    print(f"{COLOR_BLUE}[agente_tarefas_casa] Tema recebido: {tema!r}{COLOR_RESET}")
    print(f"{COLOR_BLUE}[agente_tarefas_casa] Quantidade de contextos recebidos: {len(contextos)}{COLOR_RESET}")
    print(f"{COLOR_BLUE}[agente_tarefas_casa] Modelo LLM usado: {llm_model!r}{COLOR_RESET}")

    if nomes_docs is None:
        nomes_docs = ["Documento_desconhecido.pdf"] * len(contextos)

    arquivos_unicos = sorted(set(n for n in nomes_docs if n is not None))

    # Numerar contextos com documento na primeira linha
    contextos_marcados = []
    for ctx, doc_name in zip(contextos, nomes_docs):
        bloco = f"[FONTE: {doc_name}]\n{ctx}"
        contextos_marcados.append(bloco)

    context = "\n\n---\n\n".join(contextos_marcados)

    arquivos_str = "\n".join(f"- {nome}" for nome in arquivos_unicos)

    user_prompt = f"""
Você é um professor universitário de Inteligência Artificial.

Use EXCLUSIVAMENTE os contextos abaixo para criar tarefas de casa.
Cada bloco começa com uma linha no formato:
[FONTE: NOME_DO_ARQUIVO.pdf]

ARQUIVOS DISPONÍVEIS (USE APENAS ESSES NOMES, COPIANDO-OS EXATAMENTE):
{arquivos_str if arquivos_str else "- (nenhum nome de arquivo disponível)"}

REGRAS SOBRE NOMES DE ARQUIVOS:
- Você SÓ pode citar arquivos usando exatamente um dos nomes listados acima.
- NÃO invente nomes de arquivos, abreviações, rótulos genéricos ou números.
- NÃO use "Arquivo PDF 1", "Arquivo PDF 2", "Documento X" etc.
- Se precisar citar um arquivo, escolha um nome EXATAMENTE da lista ARQUIVOS DISPONÍVEIS.
- Se a informação não puder ser associada a nenhum arquivo da lista, NÃO invente um nome — apenas explique o conteúdo sem atribuir a um arquivo específico ou diga que não há arquivo associado.

REGRAS GERAIS IMPORTANTES:
- NÃO invente teorias, algoritmos, exemplos ou cenários que NÃO estejam nos contextos fornecidos.
- Se algum tipo de exercício ou informação não puder ser bem fundamentado a partir dos materiais,
  escreva explicitamente algo como:
  "Este ponto não está claramente presente nos materiais fornecidos."
- Mantenha-se fiel aos conteúdos dos arquivos.

CONTEXTOS:
{context}

TAREFA:
Crie um conjunto de tarefas de casa (homework) alinhadas ao tema, contendo:
- 1 ou 2 exercícios teóricos (dissertativos, análise crítica, estudo de caso)
- 1 ou 2 exercícios práticos (podendo envolver IA generativa, RAG, análise de artigos etc.)
- Em cada exercício, quando fizer sentido, deixe claro entre parênteses
  qual arquivo embasou a atividade (ex.: "(baseado em Attention Is All You Need.pdf)").
- Inclua instruções claras para o aluno (o que entregar, formato, prazo sugerido)
- Inclua critérios gerais de avaliação

SEÇÃO FINAL OBRIGATÓRIA:
- "Documentos usados": liste todos os nomes de arquivos PDF que você mencionou nos exercícios.
  - Nesta lista, COPIE EXATAMENTE os nomes de ARQUIVOS DISPONÍVEIS.
  - Não invente nomes genéricos nem rótulos numéricos.

FORMATO DA RESPOSTA:
- A resposta deve ser SOMENTE o conjunto de tarefas de casa em Markdown.
- NÃO inclua frases como "Segue as tarefas:", "Como solicitado", "Conclusão" fora da estrutura normal.
- NÃO explique o que você está fazendo, apenas apresente os exercícios, instruções e critérios.

IMPORTANTE:
- Toda a resposta deve estar em português.
- Se precisar detalhar algo que não esteja explícito nos textos, deixe claro que se trata
  de uma inferência pedagógica e não de uma citação direta dos materiais.

Tema:
\"\"\"{tema}\"\"\"

Escreva em português, bem estruturado em Markdown.
"""

    print(f"{COLOR_MAGENTA}[agente_tarefas_casa] Chamando o modelo Ollama...{COLOR_RESET}")
    response = ollama.chat(
        model=llm_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Você cria tarefas universitárias de Inteligência Artificial "
                    "baseadas APENAS nos contextos fornecidos, sem inventar fatos, "
                    "NUNCA inventa nomes de arquivos e SEMPRE usa exatamente os nomes da lista de arquivos. "
                    "Sua resposta deve ser apenas o conjunto de tarefas final, sem comentários adicionais."
                ),
            },
            {"role": "user", "content": textwrap.dedent(user_prompt).strip()},
        ],
    )
    conteudo_resposta = response["message"]["content"]

    print(f"{COLOR_GREEN}[agente_tarefas_casa] Conteúdo gerado. Salvando arquivo MD...{COLOR_RESET}")

    ensure_output_dir()
    filename = os.path.join(OUTPUT_DIR, f"tarefas_de_casa_{sanitize_filename(tema)}.md")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(conteudo_resposta)

    print(f"{COLOR_GREEN}[agente_tarefas_casa] Arquivo salvo: {filename}{COLOR_RESET}")

    return conteudo_resposta
