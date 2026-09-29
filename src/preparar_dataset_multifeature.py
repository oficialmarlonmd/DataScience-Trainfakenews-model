import pandas as pd
import os

print("🔄 Iniciando Engenharia de Features (Data Prep - Multifeature)...")

# Caminhos
input_path = 'data/processed/noticias_balanceadas_expandidas_limpas.csv'
output_path = 'data/processed/dataset_treino_multifeature_v2.csv'

# Carregar dataset original
print(f"Lendo dataset base: {input_path}")
df = pd.read_csv(input_path)

# Tratamento de Nulos
df['titulo'] = df['titulo'].fillna('')
df['subtitulo'] = df['subtitulo'].fillna('')
df['texto'] = df['texto'].fillna('')

# Concatenando as features com os tokens especiais do BERT
# O token [SEP] ajuda o modelo a entender onde termina o título e onde começa o corpo
print("Costurando features (Título + Subtítulo + Texto)...")
df['texto_enriquecido'] = df['titulo'] + " [SEP] " + df['subtitulo'] + " [SEP] " + df['texto']

# Limpeza de espaços extras criados na junção
df['texto_enriquecido'] = df['texto_enriquecido'].str.replace(r'\s+', ' ', regex=True).str.strip()

# Selecionar apenas as colunas que importam para o treinamento
# A classe precisa estar mapeada como 0 (Falsa) e 1 (Verdadeira) se ainda não estiver
print("Mapeando classes e formatando dataframe final...")
df_final = pd.DataFrame()
df_final['texto'] = df['texto_enriquecido']

# Garantindo o mapeamento de classe para inteiro (0 e 1)
class_map = {'falsa': 0, 'verdadeira': 1}
if df['classe'].dtype == object:
    df_final['label'] = df['classe'].str.lower().map(class_map)
else:
    df_final['label'] = df['classe']

# Salvar o novo dataset
df_final.to_csv(output_path, index=False)
print(f"✅ Dataset V2 salvo com sucesso em: {output_path}")
print(f"📊 Amostras preparadas: {len(df_final)}")
print("\nExemplo de como o modelo vai ler a notícia a partir de agora:")
print(df_final['texto'].iloc[0][:300] + "...")
