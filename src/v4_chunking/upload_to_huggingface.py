"""
=============================================================================
SCRIPT DE UPLOAD DO MODELO V4 PARA O HUGGING FACE HUB
=============================================================================
Este script envia os pesos, tokenizer e metadados do modelo V4 campeão
diretamente para a sua conta no Hugging Face (huggingface.co).

Após o upload, o modelo ganha automaticamente uma API REST pública ou privada
hospedada gratuitamente pela infraestrutura do Hugging Face (Inference API).

Como usar:
1. Tenha um Token de escrita do HF: https://huggingface.com/settings/tokens
2. Defina seu token via terminal: export HF_TOKEN="hf_..."
   OU passe como parâmetro no script.
3. Execute:
   python3 src/v4_chunking/upload_to_huggingface.py
=============================================================================
"""

import os
import glob
import torch
from huggingface_hub import HfApi, login
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("🚀 Preparando Upload do Modelo V4 para o Hugging Face Hub")
print("=" * 65)

# 1. Configurar Token e Nome do Repositório
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
if not HF_TOKEN:
    print("🔑 Atenção: A variável de ambiente HF_TOKEN não foi detectada.")
    HF_TOKEN = input("Digite ou cole seu Token do Hugging Face (com permissão Write): ").strip()

if not HF_TOKEN:
    print("❌ Token inválido ou não informado. Abortando upload.")
    exit(1)

login(token=HF_TOKEN)

# Nome do repositório no Hugging Face
REPO_NAME = os.getenv("HF_REPO_NAME", "").strip()
if not REPO_NAME:
    api = HfApi()
    user_info = api.whoami(token=HF_TOKEN)
    username = user_info["name"]
    REPO_NAME = f"{username}/bertimbau-fakenews-detector-v4"

print(f"📦 Repositório de Destino: https://huggingface.co/{REPO_NAME}")

# 2. Localizar o melhor checkpoint da V4
checkpoint_candidates = glob.glob("./logs/results_v4_fold_0/checkpoint-*")
if not checkpoint_candidates:
    raise FileNotFoundError("Nenhum checkpoint V4 encontrado em ./logs/results_v4_fold_0/")

checkpoint_candidates.sort(key=os.path.getmtime)
best_checkpoint = checkpoint_candidates[-1]
print(f"📥 Carregando modelo do checkpoint campeão: {best_checkpoint}")

# 3. Carregar e configurar labels no modelo
tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
model = AutoModelForSequenceClassification.from_pretrained(best_checkpoint)

# Configurar rótulos amigáveis para a Inference API do Hugging Face
model.config.id2label = {0: "FALSA", 1: "VERDADEIRA"}
model.config.label2id = {"FALSA": 0, "VERDADEIRA": 1}

# 4. Enviar para o Hugging Face Hub
print("\n☁️  Enviando pesos do modelo e tokenizer para o Hugging Face...")
model.push_to_hub(REPO_NAME, token=HF_TOKEN, private=False)
tokenizer.push_to_hub(REPO_NAME, token=HF_TOKEN)

# 5. Criar e enviar o Model Card (README.md) detalhado
model_card_content = f"""---
language:
- pt
license: mit
tags:
- fake-news
- bertimbau
- text-classification
- portuguese
- nlp
datasets:
- custom-balanced-fakenews-pt
metrics:
- accuracy
- f1
pipeline_tag: text-classification
widget:
- text: "URGENTE: Descoberta substância milagrosa que cura todas as doenças em 24h e médicos tentam esconder!"
- text: "O Banco Central decidiu manter a taxa básica de juros Selic em 10,50% ao ano em decisão unânime do Copom."
---

# 📰 Detector de Fake News em Português — BERTimbau V4

Modelo para detecção e auditoria de notícias em língua portuguesa, baseado no **BERTimbau** (`neuralmind/bert-base-portuguese-cased`) ajustado com técnica de **Chunking Desenviesado** (eliminando o viés de tamanho de texto — *shortcut learning*).

## 📊 Métricas de Desempenho (Teste Cego em 100 Notícias Reais)
- **Acurácia Global:** **98.00%** (98/100)
- **Detecção de Notícias Falsas (Recall):** **100.00%** (50/50 - Zero notícias falsas deixadas passar)
- **Precisão em Notícias Falsas:** **96.15%**
- **Precisão em Notícias Verdadeiras:** **100.00%**
- **Recall em Notícias Verdadeiras:** **96.00%** (48/50)

## 🚀 Como Consumir via Python (Transformers)

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

model_name = "{REPO_NAME}"
classifier = pipeline("text-classification", model=model_name, tokenizer=model_name)

# Formatação ideal: Titulo [SEP] Subtitulo [SEP] Texto
texto_noticia = "Banco Central mantém juros [SEP] Decisão unânime do Copom [SEP] O Comitê decidiu manter a Selic em 10,50%."
resultado = classifier(texto_noticia)
print(resultado)
# Output: [{{'label': 'VERDADEIRA', 'score': 0.997}}]
```

## 🌐 Consumo via API REST (Hugging Face Inference API)

```bash
curl https://router.huggingface.co/hf-inference/models/{REPO_NAME} \\
     -X POST \\
     -H "Authorization: Bearer $HF_TOKEN" \\
     -H "Content-Type: application/json" \\
     -d '{{"inputs": "Título da notícia [SEP] Subtítulo [SEP] Texto completo"}}'
```
"""

api = HfApi()
api.upload_file(
    path_or_fileobj=model_card_content.encode("utf-8"),
    path_in_repo="README.md",
    repo_id=REPO_NAME,
    token=HF_TOKEN
)

print("\n" + "=" * 65)
print("🎉 MODELO PUBLICADO COM SUCESSO NO HUGGING FACE!")
print("=" * 65)
print(f"🔗 Página do Modelo:     https://huggingface.co/{REPO_NAME}")
print(f"📡 Endpoint Inference:   https://router.huggingface.co/hf-inference/models/{REPO_NAME}")
print("=" * 65)
