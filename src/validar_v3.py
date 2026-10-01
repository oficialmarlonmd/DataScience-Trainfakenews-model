"""
=================================================================
VALIDAÇÃO DO MODELO V3 (RECALIBRADO COM MAX_LEN=512)
Avalia o modelo nas 84+ notícias reais (dataset de validação)
e compara com a baseline anterior (V2: 41.7% ou viés de tamanho).
=================================================================
"""
import os
import glob
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("🔍 Iniciando Validação do Modelo V3 Recalibrado")
print("=" * 60)

# 1. Localizar o melhor checkpoint do V3
checkpoint_candidates = glob.glob("./logs/results_v3_fold_*/checkpoint-*")
if not checkpoint_candidates:
    print("⚠️ Nenhum checkpoint V3 encontrado ainda em ./logs/results_v3_fold_*/")
    print("Verifique se o treino V3 já completou pelo menos 1 época.")
    exit(1)

# Ordenar para pegar o mais recente ou com maior número de passos
checkpoint_candidates.sort(key=os.path.getmtime)
best_checkpoint = checkpoint_candidates[-1]
print(f"📥 Carregando modelo do checkpoint: {best_checkpoint}")

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Dispositivo de inferência: {device}")

model = AutoModelForSequenceClassification.from_pretrained(best_checkpoint)
model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")

# 2. Carregar dataset de validação com notícias reais
val_path = "data/processed/validacao_100_multifeature.csv"
if not os.path.exists(val_path):
    val_path = "data/processed/validacao_multifeature.csv"

print(f"📖 Lendo dataset de validação: {val_path}")
df_val = pd.read_csv(val_path)

df_val['titulo'] = df_val['titulo'].fillna('')
df_val['subtitulo'] = df_val['subtitulo'].fillna('')
df_val['texto'] = df_val['texto'].fillna('')

class_map = {'falsa': 0, 'verdadeira': 1, '0': 0, '1': 1, 0: 0, 1: 1}
df_val['target'] = df_val['classe_real'].astype(str).str.lower().map(class_map)

# Prepara os textos com formato multifeature
df_val['texto_input'] = df_val['titulo'] + " [SEP] " + df_val['subtitulo'] + " [SEP] " + df_val['texto']
df_val['texto_input'] = df_val['texto_input'].str.replace(r'\s+', ' ', regex=True).str.strip()

total_amostras = len(df_val)
total_falsas = (df_val['target'] == 0).sum()
total_verdadeiras = (df_val['target'] == 1).sum()

print(f"⚖️ Dataset de validação contém {total_amostras} notícias:")
print(f"   • Falsas reais: {total_falsas}")
print(f"   • Verdadeiras reais: {total_verdadeiras}")
print("-" * 60)

acertos = 0
falsas_corretas = 0
verdadeiras_corretas = 0
predicoes = []
probs_verd = []

for idx, row in df_val.iterrows():
    text = row['texto_input']
    real_label = row['target']
    
    # MAX_LEN=512 conforme calibrado no V3
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze().cpu().numpy()
        pred = np.argmax(probs)
        
    predicoes.append(pred)
    probs_verd.append(probs[1])
    
    if pred == real_label:
        acertos += 1
        if real_label == 0:
            falsas_corretas += 1
        else:
            verdadeiras_corretas += 1

acc = (acertos / total_amostras) * 100
prec_falsas = (falsas_corretas / total_falsas) * 100 if total_falsas > 0 else 0
prec_verdadeiras = (verdadeiras_corretas / total_verdadeiras) * 100 if total_verdadeiras > 0 else 0

print("\n" + "=" * 60)
print("🏆 RESULTADO DA VALIDAÇÃO — MODELO V3 (MAX_LEN=512)")
print("=" * 60)
print(f"Acurácia Geral:          {acertos}/{total_amostras} ({acc:.2f}%)")
print(f"Acerto em Falsas:       {falsas_corretas}/{total_falsas} ({prec_falsas:.2f}%)")
print(f"Acerto em Verdadeiras:  {verdadeiras_corretas}/{total_verdadeiras} ({prec_verdadeiras:.2f}%)")
print("-" * 60)
print(f"Predições como 'Falsa':       {(np.array(predicoes) == 0).sum()}")
print(f"Predições como 'Verdadeira': {(np.array(predicoes) == 1).sum()}")
print(f"Média Probabilidade Verdadeira: {np.mean(probs_verd):.3f}")

if (np.array(predicoes) == 1).sum() == total_amostras:
    print("🚨 ALERTA: O modelo V3 ainda classificou 100% das notícias como 'Verdadeira'!")
    print("   Isso confirma que aumentar MAX_LEN para 512 NÃO removeu o viés de tamanho.")
    print("   Avançar imediatamente para o V4 (Chunking / Fatiamento).")
elif acc > 75.0:
    print("🎉 SUCESSO: O modelo V3 conseguiu distinguir as classes no teste real!")
else:
    print("ℹ️ O modelo teve comportamento misto. Analisar detalhadamente os erros.")
