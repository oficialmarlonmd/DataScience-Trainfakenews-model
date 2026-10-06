"""
=============================================================================
EXEMPLO DE CONSUMO DA API DO HUGGING FACE (INFERENCE API)
=============================================================================
Como chamar o modelo hospedado no Hugging Face via Python / HTTP.
=============================================================================
"""

import os
from transformers import pipeline

REPO_NAME = os.getenv("HF_REPO_NAME", "oficialmarlon/bertimbau-fakenews-detector-v4")
HF_TOKEN = os.getenv("HF_TOKEN", "")

print(f"📡 Carregando modelo do Hugging Face: {REPO_NAME}")

# Inicializa o pipeline direto do Hugging Face
classifier = pipeline(
    "text-classification",
    model=REPO_NAME,
    token=HF_TOKEN if HF_TOKEN else None
)

def verificar_noticia(titulo, subtitulo="", texto=""):
    entrada = f"{titulo} [SEP] {subtitulo} [SEP] {texto}".strip()
    resultado = classifier(entrada)
    return resultado[0]

# Teste 1: Notícia Falsa
print("\n--- Teste 1: Notícia Falsa ---")
res_fake = verificar_noticia(
    titulo="URGENTE: Nova substância milagrosa cura todas as doenças em 24h!",
    subtitulo="Médicos tentam esconder a receita secreta da população.",
    texto="Compartilhe imediatamente antes que derrubem este artigo."
)
print("Resultado:", res_fake)

# Teste 2: Notícia Verdadeira
print("\n--- Teste 2: Notícia Verdadeira ---")
res_true = verificar_noticia(
    titulo="Banco Central mantém taxa básica de juros Selic em 10,50% ao ano",
    subtitulo="Decisão foi unânime pelo Comitê de Política Monetária (Copom).",
    texto="O Comitê de Política Monetária do Banco Central decidiu por unanimidade manter a taxa Selic."
)
print("Resultado:", res_true)

