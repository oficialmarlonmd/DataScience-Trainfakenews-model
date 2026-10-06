"""
=============================================================================
EXEMPLO DE CONSUMO DA API DO HUGGING FACE (INFERENCE API)
=============================================================================
Como chamar o modelo hospedado no Hugging Face via Python / HTTP.
=============================================================================
"""

import os
import requests

# Substitua pelo seu repo ou defina a variável HF_REPO_NAME
REPO_NAME = os.getenv("HF_REPO_NAME", "seu-usuario/bertimbau-fakenews-detector-v4")
HF_TOKEN = os.getenv("HF_TOKEN", "")

API_URL = f"https://router.huggingface.co/hf-inference/models/{REPO_NAME}"
headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

payload = {
    "inputs": "URGENTE: Remédio caseiro milagroso cura câncer e médicos estão chocados! [SEP] Veja a receita secreta [SEP] Médicos tentam proibir a divulgação deste chá natural."
}

response = requests.post(API_URL, headers=headers, json=payload)

if response.status_code == 200:
    print("Predição do Modelo no Hugging Face:")
    print(response.json())
else:
    print(f"Status {response.status_code}:", response.text)
