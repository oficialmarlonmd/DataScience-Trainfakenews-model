"""
=================================================================
INVESTIGAÇÃO FORENSE DO MODELO V2
Diagnóstico: Por que o modelo classifica tudo como 'verdadeira'?
=================================================================
"""
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("🔬 INVESTIGAÇÃO FORENSE DO MODELO V2")
print("=" * 60)

# 1. ANÁLISE DO DATASET DE TREINO
print("\n📊 ETAPA 1: Análise de Distribuição do Dataset de Treino")
print("-" * 60)
df_treino = pd.read_csv("data/processed/dataset_treino_multifeature_v2.csv")
df_treino['label'] = df_treino['label'].map({'falsa': 0, 'verdadeira': 1, 0: 0, 1: 1, '0': 0, '1': 1})
df_treino['comprimento'] = df_treino['texto'].str.len()
df_treino['num_tokens_aprox'] = df_treino['texto'].str.split().str.len()

print(f"Total de amostras: {len(df_treino)}")
print(f"  Falsas (0): {(df_treino['label']==0).sum()}")
print(f"  Verdadeiras (1): {(df_treino['label']==1).sum()}")

print(f"\nComprimento médio (caracteres):")
print(f"  Falsas:      {df_treino[df_treino['label']==0]['comprimento'].mean():.0f} chars")
print(f"  Verdadeiras: {df_treino[df_treino['label']==1]['comprimento'].mean():.0f} chars")
print(f"  Ratio (Verd/Falsa): {df_treino[df_treino['label']==1]['comprimento'].mean() / df_treino[df_treino['label']==0]['comprimento'].mean():.2f}x")

print(f"\nNúmero médio de palavras:")
print(f"  Falsas:      {df_treino[df_treino['label']==0]['num_tokens_aprox'].mean():.0f} palavras")
print(f"  Verdadeiras: {df_treino[df_treino['label']==1]['num_tokens_aprox'].mean():.0f} palavras")

# Quantas amostras ficam com mais de 128 tokens (MAX_LEN do treino)?
print(f"\n⚠️ MAX_LEN usado no treino: 128 tokens")
print(f"  Falsas com > 128 palavras:      {(df_treino[df_treino['label']==0]['num_tokens_aprox'] > 128).sum()} / {(df_treino['label']==0).sum()}")
print(f"  Verdadeiras com > 128 palavras: {(df_treino[df_treino['label']==1]['num_tokens_aprox'] > 128).sum()} / {(df_treino['label']==1).sum()}")

# 2. ANÁLISE DOS LOGITS DO MODELO
print(f"\n📊 ETAPA 2: Análise de Confiança do Modelo (Logits e Probabilidades)")
print("-" * 60)

model = AutoModelForSequenceClassification.from_pretrained("./logs/results_fold_0/checkpoint-760")
model.eval()
tokenizer = AutoTokenizer.from_pretrained("./logs/results_fold_0/checkpoint-760")

# Testar com o dataset de validação multifeature
df_val = pd.read_csv("data/processed/validacao_multifeature.csv")
df_val['titulo'] = df_val['titulo'].fillna('')
df_val['subtitulo'] = df_val['subtitulo'].fillna('')
df_val['texto'] = df_val['texto'].fillna('')
df_val['texto_enriquecido'] = df_val['titulo'] + " [SEP] " + df_val['subtitulo'] + " [SEP] " + df_val['texto']

class_map = {'falsa': 0, 'verdadeira': 1}

print(f"\n{'Notícia':<50} | {'Real':<10} | {'Pred':<10} | {'Logit[Falsa]':>12} | {'Logit[Verd]':>12} | {'Prob[Falsa]':>11} | {'Prob[Verd]':>11}")
print("-" * 165)

for idx, row in df_val.iterrows():
    text = row['texto_enriquecido']
    real = row['classe_real']
    
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(logits).item()
    
    pred_str = "verdadeira" if pred == 1 else "falsa"
    titulo_curto = row['titulo'][:48]
    
    print(f"{titulo_curto:<50} | {real:<10} | {pred_str:<10} | {logits[0].item():>12.4f} | {logits[1].item():>12.4f} | {probs[0].item():>10.4f}% | {probs[1].item():>10.4f}%")

# 3. TESTE COM DIFERENTES MAX_LEN
print(f"\n\n📊 ETAPA 3: Impacto do MAX_LEN na Predição")
print("-" * 60)
print("Testando a mesma notícia FALSA com diferentes truncamentos:\n")

fake_text = df_val[df_val['classe_real'] == 'falsa'].iloc[0]['texto_enriquecido']
fake_titulo = df_val[df_val['classe_real'] == 'falsa'].iloc[0]['titulo'][:50]

for max_len in [32, 64, 128, 256, 512]:
    inputs = tokenizer(fake_text, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits[0], dim=-1)
        pred = torch.argmax(outputs.logits[0]).item()
    pred_str = "verdadeira" if pred == 1 else "falsa"
    n_tokens = inputs['attention_mask'].sum().item()
    print(f"  MAX_LEN={max_len:>3} | Tokens usados: {n_tokens:>3} | Pred: {pred_str:<12} | Prob[Falsa]={probs[0].item():.4f} | Prob[Verd]={probs[1].item():.4f}")

# 4. TESTE: TRUNCANDO SÓ O TÍTULO (sem texto longo)
print(f"\n\n📊 ETAPA 4: Predição usando SOMENTE o Título (sem texto longo)")
print("-" * 60)
print("Se o modelo é viciado em comprimento, ao dar SÓ o título ele deve mudar a predição:\n")

for idx, row in df_val.iterrows():
    title_only = row['titulo']
    real = row['classe_real']
    
    inputs = tokenizer(title_only, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits[0], dim=-1)
        pred = torch.argmax(outputs.logits[0]).item()
    
    pred_str = "verdadeira" if pred == 1 else "falsa"
    titulo_curto = row['titulo'][:55]
    print(f"  {titulo_curto:<57} | Real: {real:<10} | Pred: {pred_str:<10} | P(F)={probs[0].item():.3f} P(V)={probs[1].item():.3f}")

print("\n\n🏁 CONCLUSÃO DA INVESTIGAÇÃO")
print("=" * 60)
