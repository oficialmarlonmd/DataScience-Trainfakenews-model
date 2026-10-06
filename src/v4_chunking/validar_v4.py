"""
=================================================================
VALIDAÇÃO DO MODELO V4 (TREINADO COM CHUNKING)
Avalia o modelo V4 nas 84+ notícias reais e compara diretamente
com o baseline V2 (41.7%) e com o modelo V3.
=================================================================
"""
import os
import glob
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("🔍 Iniciando Validação do Modelo V4 (Chunking / Desenviesado)")
print("=" * 60)

# Localizar o melhor checkpoint do V4
checkpoint_candidates = glob.glob("./logs/results_v4_fold_*/checkpoint-*")
if not checkpoint_candidates:
    print("⚠️ Nenhum checkpoint V4 encontrado em ./logs/results_v4_fold_*/")
    print("Execute o treino com: python3 src/train_v4_chunking.py")
    exit(1)

checkpoint_candidates.sort(key=os.path.getmtime)
best_checkpoint = checkpoint_candidates[-1]
print(f"📥 Carregando modelo do checkpoint: {best_checkpoint}")

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Dispositivo de inferência: {device}")

model = AutoModelForSequenceClassification.from_pretrained(best_checkpoint)
model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")

val_path = "data/processed/validacao_100_multifeature.csv"
if not os.path.exists(val_path):
    val_path = "data/processed/validacao_multifeature.csv"

df_val = pd.read_csv(val_path)
df_val['titulo'] = df_val['titulo'].fillna('')
df_val['subtitulo'] = df_val['subtitulo'].fillna('')
df_val['texto'] = df_val['texto'].fillna('')

class_map = {'falsa': 0, 'verdadeira': 1, '0': 0, '1': 1, 0: 0, 1: 1}
df_val['target'] = df_val['classe_real'].astype(str).str.lower().map(class_map)

df_val['texto_input'] = df_val['titulo'] + " [SEP] " + df_val['subtitulo'] + " [SEP] " + df_val['texto']
df_val['texto_input'] = df_val['texto_input'].str.replace(r'\s+', ' ', regex=True).str.strip()

total_amostras = len(df_val)
total_falsas = (df_val['target'] == 0).sum()
total_verdadeiras = (df_val['target'] == 1).sum()

acertos = 0
falsas_corretas = 0
verdadeiras_corretas = 0
predicoes = []
probs_verd = []
erros = []

for idx, row in df_val.iterrows():
    text = row['texto_input']
    real_label = row['target']
    
    # MAX_LEN=128 conforme calibrado no V4
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
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
    else:
        tipo_erro = "Falsa prevista como Verdadeira (Falso Negativo)" if real_label == 0 else "Verdadeira prevista como Falsa (Falso Positivo)"
        erros.append({
            "titulo": row['titulo'][:50],
            "tipo": tipo_erro,
            "prob_verdadeira": probs[1]
        })

acc = (acertos / total_amostras) * 100
prec_falsas = (falsas_corretas / total_falsas) * 100 if total_falsas > 0 else 0
prec_verdadeiras = (verdadeiras_corretas / total_verdadeiras) * 100 if total_verdadeiras > 0 else 0

print("\n" + "=" * 60)
print("🏆 RESULTADO DA VALIDAÇÃO — MODELO V4 (CHUNKING)")
print("=" * 60)
print(f"Acurácia Geral:          {acertos}/{total_amostras} ({acc:.2f}%)")
print(f"Acerto em Falsas:       {falsas_corretas}/{total_falsas} ({prec_falsas:.2f}%)")
print(f"Acerto em Verdadeiras:  {verdadeiras_corretas}/{total_verdadeiras} ({prec_verdadeiras:.2f}%)")
print("-" * 60)
print(f"Predições como 'Falsa':       {(np.array(predicoes) == 0).sum()}")
print(f"Predições como 'Verdadeira': {(np.array(predicoes) == 1).sum()}")
print(f"Média Probabilidade Verdadeira: {np.mean(probs_verd):.3f}")

if erros:
    print(f"\n⚠️ Detalhamento dos Erros ({len(erros)} notícias erradas):")
    for i, e in enumerate(erros, 1):
        print(f"  {i}. {e['titulo']}... -> {e['tipo']} (Prob[Verd]: {e['prob_verdadeira']:.2%})")
else:
    print("\n🎉 GABARITOU! 100% de acerto em todas as notícias!")
