"""
=================================================================
FASE 4.6: ENGENHARIA DE FEATURES V4 - CHUNKING (FATIAMENTO)
Objetivo:
  Eliminar o viés de comprimento dividindo as notícias verdadeiras
  (longas) em chunks de ~35 palavras com título e subtítulo preservados,
  igualando a média de palavras das Verdadeiras (~49) com as Falsas (~51)
  sem perder semântica e mantendo balanceamento 1:1 rigoroso.
=================================================================
"""
import pandas as pd
import numpy as np
import os

print("🔄 Iniciando Preparação do Dataset V4 (Chunking & Rebalanceamento)...")

input_path = 'data/processed/noticias_balanceadas_expandidas_limpas.csv'
output_path = 'data/processed/dataset_treino_chunking_v4.csv'

print(f"📖 Lendo dataset base: {input_path}")
df = pd.read_csv(input_path)

df['titulo'] = df['titulo'].fillna('')
df['subtitulo'] = df['subtitulo'].fillna('')
df['texto'] = df['texto'].fillna('')

def chunk_text(text, target_words=35):
    """
    Divide um texto longo em fatias de aproximadamente target_words.
    Garante que não sobre uma fatia minúscula (<15 palavras) unindo-a à anterior.
    """
    words = text.split()
    if len(words) <= target_words:
        return [text]
    
    chunks = []
    for i in range(0, len(words), target_words):
        chunk = " ".join(words[i:i+target_words])
        if len(chunk.split()) >= 15 or len(chunks) == 0:
            chunks.append(chunk)
        else:
            chunks[-1] = chunks[-1] + " " + chunk
    return chunks

rows = []
for idx, row in df.iterrows():
    classe = str(row['classe']).lower()
    titulo = str(row['titulo']).strip()
    subtitulo = str(row['subtitulo']).strip()
    texto = str(row['texto']).strip()
    
    if classe == 'verdadeira' or classe == '1':
        chunks = chunk_text(texto, target_words=35)
        for c in chunks:
            full_text = f"{titulo} [SEP] {subtitulo} [SEP] {c}".strip()
            # Limpeza de espaços extras
            full_text = " ".join(full_text.split())
            rows.append({'texto': full_text, 'label': 1, 'origem_idx': idx})
    else:
        full_text = f"{titulo} [SEP] {subtitulo} [SEP] {texto}".strip()
        full_text = " ".join(full_text.split())
        rows.append({'texto': full_text, 'label': 0, 'origem_idx': idx})

df_chunked = pd.DataFrame(rows)

# Balanceamento perfeito 1:1
df_falsas = df_chunked[df_chunked['label'] == 0]
n_falsas = len(df_falsas)
df_verdadeiras = df_chunked[df_chunked['label'] == 1].sample(n=n_falsas, random_state=42)

df_final = pd.concat([df_falsas, df_verdadeiras]).sample(frac=1.0, random_state=42).reset_index(drop=True)
df_final['num_words'] = df_final['texto'].str.split().str.len()

# Salvar arquivo versionado V4
df_final[['texto', 'label']].to_csv(output_path, index=False)

print(f"\n✅ Dataset V4 salvo com sucesso em: {output_path}")
print("📊 Estatísticas Finais de Distribuição:")
print(f"  Total de Amostras: {len(df_final)} ({n_falsas} Falsas / {len(df_verdadeiras)} Verdadeiras)")
print(f"  Falsas:      Média={df_final[df_final['label']==0]['num_words'].mean():.1f} palavras | Mediana={df_final[df_final['label']==0]['num_words'].median():.1f}")
print(f"  Verdadeiras: Média={df_final[df_final['label']==1]['num_words'].mean():.1f} palavras | Mediana={df_final[df_final['label']==1]['num_words'].median():.1f}")
print(f"  Ratio de Comprimento: {df_final[df_final['label']==1]['num_words'].mean() / df_final[df_final['label']==0]['num_words'].mean():.2f}x (Praticamente 1:1! 🎯)")

print("\nExemplo de amostra fatiada:")
print(df_final['texto'].iloc[0][:200] + "...")
