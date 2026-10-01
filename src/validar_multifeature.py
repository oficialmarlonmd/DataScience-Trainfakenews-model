import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("🔍 Iniciando Validação Estrutural (Dataset MultiFeature)")

print("📥 Carregando modelo do HuggingFace (V2)...")
model = AutoModelForSequenceClassification.from_pretrained("./logs/results_fold_0/checkpoint-760")
model.eval()

print("📥 Carregando tokenizador...")
tokenizer = AutoTokenizer.from_pretrained("./logs/results_fold_0/checkpoint-760")

df = pd.read_csv("data/processed/validacao_multifeature.csv")
df['titulo'] = df['titulo'].fillna('')
df['subtitulo'] = df['subtitulo'].fillna('')
df['texto'] = df['texto'].fillna('')

# 🧩 Aplicando a mesmíssima costura feita na Fase 3
df['texto_enriquecido'] = df['titulo'] + " [SEP] " + df['subtitulo'] + " [SEP] " + df['texto']

class_map = {'falsa': 0, 'verdadeira': 1}

acertos = 0
falsas_corretas = 0
verdadeiras_corretas = 0
total_falsas = sum(df['classe_real'] == 'falsa')
total_verdadeiras = sum(df['classe_real'] == 'verdadeira')

print(f"⚖️ Dataset estruturado contém {total_verdadeiras} Verdadeiras e {total_falsas} Falsas.\n")
print("Processando previsões corretas...\n")

for idx, row in df.iterrows():
    text = row['texto_enriquecido']
    real_label = class_map[row['classe_real'].lower()]
    
    # Prepara a entrada
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    
    # Infere
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1).item()
        
    # Exibir log da predição individual para provar que funciona
    pred_str = "verdadeira" if prediction == 1 else "falsa"
    print(f"Notícia: {row['titulo'][:40]}... | Correto: {row['classe_real']} | Predição: {pred_str}")
        
    # Conta
    if prediction == real_label:
        acertos += 1
        if real_label == 0:
            falsas_corretas += 1
        else:
            verdadeiras_corretas += 1

print("\n🏆 --- RESULTADO FINAL (MODELO V2 MULTIFEATURE) --- 🏆")
print(f"Acurácia Geral: {acertos}/{len(df)} ({acertos/len(df)*100:.1f}%)")
print(f"Precisão em Falsas: {falsas_corretas}/{total_falsas} ({(falsas_corretas/total_falsas)*100:.1f}%)")
print(f"Precisão em Verdadeiras: {verdadeiras_corretas}/{total_verdadeiras} ({(verdadeiras_corretas/total_verdadeiras)*100:.1f}%)")
