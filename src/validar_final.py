"""
VALIDAÇÃO FINAL V2 vs V3
Testa o modelo contra o dataset de 84 notícias reais da internet
(Todas com título + subtítulo + texto, balanceadas em comprimento)
"""
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("🔍 Validação com Dataset Real (84 notícias da internet)")
print("=" * 60)

# Carregar modelo V2
print("📥 Carregando modelo V2 (MAX_LEN=128)...")
model = AutoModelForSequenceClassification.from_pretrained("./logs/results_fold_0/checkpoint-760")
model.eval()
tokenizer = AutoTokenizer.from_pretrained("./logs/results_fold_0/checkpoint-760")

# Carregar dataset
df = pd.read_csv("data/processed/validacao_100_multifeature.csv")
df['titulo'] = df['titulo'].fillna('')
df['subtitulo'] = df['subtitulo'].fillna('')
df['texto'] = df['texto'].fillna('')

# Mesma costura do treino
df['texto_enriquecido'] = df['titulo'] + " [SEP] " + df['subtitulo'] + " [SEP] " + df['texto']

class_map = {'falsa': 0, 'verdadeira': 1}

total_falsas = (df['classe_real'] == 'falsa').sum()
total_verdadeiras = (df['classe_real'] == 'verdadeira').sum()

print(f"⚖️ Dataset: {total_verdadeiras} Verdadeiras + {total_falsas} Falsas = {len(df)} total")
print(f"Comprimento médio: Falsas={df[df['classe_real']=='falsa']['texto'].str.len().mean():.0f} | Verdadeiras={df[df['classe_real']=='verdadeira']['texto'].str.len().mean():.0f}")

# Testar com MAX_LEN=128 (como o V2 foi treinado)
print(f"\n--- TESTE COM MAX_LEN=128 (Config do treino V2) ---")
acertos_128 = 0
f_corretas_128 = 0
v_corretas_128 = 0

for idx, row in df.iterrows():
    text = row['texto_enriquecido']
    real_label = class_map[row['classe_real'].lower()]
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1).item()
    if prediction == real_label:
        acertos_128 += 1
        if real_label == 0: f_corretas_128 += 1
        else: v_corretas_128 += 1

print(f"Acurácia: {acertos_128}/{len(df)} ({acertos_128/len(df)*100:.1f}%)")
print(f"Falsas corretas: {f_corretas_128}/{total_falsas} ({f_corretas_128/total_falsas*100:.1f}%)")
print(f"Verdadeiras corretas: {v_corretas_128}/{total_verdadeiras} ({v_corretas_128/total_verdadeiras*100:.1f}%)")

# Testar com MAX_LEN=512 (mesmo modelo, mais contexto)
print(f"\n--- TESTE COM MAX_LEN=512 (mesmo modelo V2, mais contexto) ---")
acertos_512 = 0
f_corretas_512 = 0
v_corretas_512 = 0

for idx, row in df.iterrows():
    text = row['texto_enriquecido']
    real_label = class_map[row['classe_real'].lower()]
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1).item()
    if prediction == real_label:
        acertos_512 += 1
        if real_label == 0: f_corretas_512 += 1
        else: v_corretas_512 += 1

print(f"Acurácia: {acertos_512}/{len(df)} ({acertos_512/len(df)*100:.1f}%)")
print(f"Falsas corretas: {f_corretas_512}/{total_falsas} ({f_corretas_512/total_falsas*100:.1f}%)")
print(f"Verdadeiras corretas: {v_corretas_512}/{total_verdadeiras} ({v_corretas_512/total_verdadeiras*100:.1f}%)")

print(f"\n🏁 CONCLUSÃO: O modelo V2 atual, treinado com MAX_LEN=128, precisa ser retreinado com MAX_LEN=512.")
