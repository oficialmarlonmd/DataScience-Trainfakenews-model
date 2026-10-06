"""
=============================================================================
EXPORTAÇÃO DO MODELO V3 (RECALIBRADO - MAX_LEN 512) PARA O BENTOML
=============================================================================
Registra o modelo V3 no BentoML Model Store para histórico de versões
e comparabilidade no pipeline de MLOps.
=============================================================================
"""

import os
import glob
import torch
import bentoml
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("📦 Iniciando Exportação do Modelo V3 (Recalibrado)")
print("=" * 65)

# 1. Localizar o melhor checkpoint da V3
checkpoint_candidates = glob.glob("./logs/results_v3_fold_0/checkpoint-*")
if not checkpoint_candidates:
    raise FileNotFoundError("Nenhum checkpoint V3 encontrado em ./logs/results_v3_fold_0/")

checkpoint_candidates.sort(key=os.path.getmtime)
best_checkpoint = checkpoint_candidates[-1]
print(f"📥 Carregando pesos do checkpoint V3: {best_checkpoint}")

# 2. Carregar Modelo
tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
model = AutoModelForSequenceClassification.from_pretrained(best_checkpoint)

# 3. Registrar no BentoML Model Store
print("\n📦 Registrando no BentoML Model Store...")
bento_model = bentoml.pytorch.save_model(
    "fakenews_bert_v3",
    model,
    signatures={"__call__": {"batchable": True, "batch_dim": 0}},
    labels={
        "framework": "pytorch",
        "task": "sequence-classification",
        "version": "v3-recalibrado",
        "environment": "archive"
    },
    metadata={
        "base_model": "neuralmind/bert-base-portuguese-cased",
        "max_length": 512,
        "acuracia_geral": 0.84,
        "acuracia_falsas": 0.68,
        "acuracia_verdadeiras": 1.00,
        "classes": {0: "falsa", 1: "verdadeira"},
        "checkpoint_origem": best_checkpoint
    }
)

print("\n" + "=" * 65)
print("✅ MODELO V3 EXPORTADO COM SUCESSO NO BENTOML!")
print("=" * 65)
print(f"🔖 BentoML Model Tag: {bento_model.tag}")
print("=================================================================")
print("Para listar os modelos, execute:")
print("  python3 -m bentoml models list")
print("=================================================================")
