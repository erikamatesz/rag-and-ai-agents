import textwrap
from typing import List

import ollama

LLM_MODEL_DEFAULT = "llama3"


# =========================
# AGENTE 1 – PREPARA AULA
# =========================

def agente_prepara_aula(
    tema: str,
    contextos: List[str],
    llm_model: str = LLM_MODEL_DEFAULT,
) -> str:
    """
    Usa os contextos recuperados para gerar um plano de aula
    universitário sobre IA.
    """
    context = "\n\n---\n\n".join(contextos)

    user_prompt = f"""
Você é um professor universitário de Inteligência Artificial.

Use EXCLUSIVAMENTE o contexto abaixo (que são trechos de materiais
do próprio professor sobre IA) para preparar um plano de aula
universitária.

CONTEXTOS:
{context}

TAREFA:
Crie um plano de aula com:
- Título da aula
- Objetivos de aprendizagem
- Pré-requisitos
- Conteúdos principais
- Metodologia (como conduzir a aula, tempo aproximado)
- Atividades em sala (incluindo, se fizer sentido, uso de IA generativa)
- Leituras obrigatórias e complementares (faça referência aos conceitos do contexto)
- Formas de avaliação (ex.: participação, exercícios, mini-projeto)

Tema/pedido do professor:
\"\"\"{tema}\"\"\"

Escreva em português, bem organizado com títulos e subtítulos.
"""

    response = ollama.chat(
        model=llm_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Você é um agente pedagógico especialista em ensino "
                    "universitário de Inteligência Artificial."
                ),
            },
            {"role": "user", "content": textwrap.dedent(user_prompt).strip()},
        ],
    )

    return response["message"]["content"]


# =========================
# AGENTE 2 – TAREFAS DE CASA
# =========================

def agente_tarefas_casa(
    tema: str,
    contextos: List[str],
    llm_model: str = LLM_MODEL_DEFAULT,
) -> str:
    """
    Usa os mesmos contextos recuperados para criar tarefas de casa
    (homework) alinhadas com a aula.
    """
    context = "\n\n---\n\n".join(contextos)

    user_prompt = f"""
Você é um professor universitário de Inteligência Artificial.

Use EXCLUSIVAMENTE o contexto abaixo (trechos de materiais do professor)
para criar tarefas de casa para os alunos.

CONTEXTOS:
{context}

TAREFA:
Crie um conjunto de tarefas de casa (homework) alinhadas ao tema abaixo
e ao nível universitário, incluindo:

- 3 a 5 exercícios teóricos (pergunta discursiva, estudo de caso, análise crítica)
- 2 a 3 exercícios práticos (podem envolver uso de IA generativa, RAG, análise de artigos, etc.)
- Instruções claras para o aluno (o que entregar, formato, prazo sugerido)
- Critérios gerais de avaliação (o que será valorizado nessas tarefas)

Tema/pedido do professor:
\"\"\"{tema}\"\"\"

Escreva em português, de forma clara e bem estruturada.
"""

    response = ollama.chat(
        model=llm_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Você é um agente pedagógico focado em criar tarefas de casa "
                    "para disciplinas universitárias de Inteligência Artificial."
                ),
            },
            {"role": "user", "content": textwrap.dedent(user_prompt).strip()},
        ],
    )

    return response["message"]["content"]
