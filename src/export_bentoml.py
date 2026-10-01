import bentoml
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

print("📦 Iniciando o empacotamento do Modelo (Fase 5 - MLOps)")
MODEL_CHECKPOINT_PATH = "./logs/results_fold_0/checkpoint-760"

print(f"Carregando pesos do checkpoint: {MODEL_CHECKPOINT_PATH}...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT_PATH)

print("Empacotando com BentoML (PyTorch API)...")
bento_model = bentoml.pytorch.save_model(
    "fakenews_bert_v2",
    model,
    signatures={"__call__": {"batchable": True, "batch_dim": 0}},
    metadata={"framework": "PyTorch", "base_model": "neuralmind/bert-base-portuguese-cased"}
)

print("\n✅ Sucesso! O modelo foi empacotado no BentoML.")
print(f"🔖 Tag do Modelo: {bento_model.tag}")
print("Para ver seus modelos salvos, rode no terminal: bentoml models list")
