"""
=============================================================================
EXPORTAÇÃO DO MODELO V4 CAMPEÃO PARA MLOPS E SERVING (BENTOML + STANDALONE)
=============================================================================
Exporta o checkpoint final do Modelo V4 (98% de acurácia, 100% recall em fakes)
para:
1. Diretório de Produção Local: models/bert_fakenews_v4/ (Offline & Standalone)
2. BentoML Model Store: fakenews_bert_v4 (Para Model Serving e API REST)
=============================================================================
"""

import os
import glob
import torch
import bentoml
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("🚀 Iniciando Exportação do Modelo V4 Campeão (Fase 5 - MLOps)")
print("=" * 65)

# 1. Localizar o melhor checkpoint da V4
checkpoint_candidates = glob.glob("./logs/results_v4_fold_0/checkpoint-*")
if not checkpoint_candidates:
    raise FileNotFoundError("Nenhum checkpoint V4 encontrado em ./logs/results_v4_fold_0/")

checkpoint_candidates.sort(key=os.path.getmtime)
best_checkpoint = checkpoint_candidates[-1]
print(f"📥 Carregando pesos do checkpoint campeão: {best_checkpoint}")

# 2. Carregar Modelo e Tokenizer
tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
model = AutoModelForSequenceClassification.from_pretrained(best_checkpoint)

# 3. Exportar para pasta standalone de produção (permite carregar offline)
OUTPUT_DIR = "models/bert_fakenews_v4"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"\n💾 Salvando artefatos standalone em: {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("   - Pesos do modelo (.safetensors / pytorch_model.bin) salvos.")
print("   - Configuração e Vocabulário do Tokenizer salvos.")

# 4. Registrar no Model Store do BentoML
print("\n📦 Registrando no BentoML Model Store...")
bento_model = bentoml.pytorch.save_model(
    "fakenews_bert_v4",
    model,
    signatures={"__call__": {"batchable": True, "batch_dim": 0}},
    labels={
        "framework": "pytorch",
        "task": "sequence-classification",
        "version": "v4-chunking",
        "environment": "production"
    },
    metadata={
        "base_model": "neuralmind/bert-base-portuguese-cased",
        "max_length": 128,
        "acuracia_geral": 0.98,
        "acuracia_falsas": 1.00,
        "acuracia_verdadeiras": 0.96,
        "classes": {0: "falsa", 1: "verdadeira"},
        "checkpoint_origem": best_checkpoint
    }
)

print("\n" + "=" * 65)
print("✅ MODELO V4 EXPORTADO COM SUCESSO!")
print("=" * 65)
print(f"🔖 BentoML Model Tag:   {bento_model.tag}")
print(f"📁 Pasta Standalone:    {OUTPUT_DIR}")
print("=" * 65)
print("Para listar modelos registrados no BentoML:")
print("  python3 -m bentoml models list")
print("=" * 65)
