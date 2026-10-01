import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

print("📊 Iniciando Análise Exploratória de Dados (EDA)...")

# Caminhos
dataset_path = 'data/processed/noticias_balanceadas_expandidas_limpas.csv'
output_dir = 'logs'
os.makedirs(output_dir, exist_ok=True)

# 1. Carregar Dados
print(f"Lendo o arquivo: {dataset_path}")
df = pd.read_csv(dataset_path)

# 2. Criar Features Temporárias de Comprimento
print("Calculando o comprimento (em tokens/palavras) dos textos...")
# Preencher NaNs com string vazia para não dar erro
df['titulo'] = df['titulo'].fillna('')
df['subtitulo'] = df['subtitulo'].fillna('')
df['texto'] = df['texto'].fillna('')

df['len_titulo'] = df['titulo'].apply(lambda x: len(str(x).split()))
df['len_subtitulo'] = df['subtitulo'].apply(lambda x: len(str(x).split()))
df['len_texto_completo'] = df['texto'].apply(lambda x: len(str(x).split()))
df['len_total'] = df['len_titulo'] + df['len_subtitulo'] + df['len_texto_completo']

# 3. Análise de "Sentimento Simples" (Sensacionalismo)
# Em NLP básico, caixa alta e exclamação gritam "Sensacionalismo" (típico de Fakes)
df['qtd_exclamacoes'] = df['titulo'].apply(lambda x: str(x).count('!'))
df['qtd_caixa_alta'] = df['titulo'].apply(lambda x: sum(1 for c in str(x) if c.isupper()))

# 4. Plotar Distribuição de Comprimento de Texto (Verdadeiras vs Falsas)
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='len_texto_completo', hue='classe', bins=50, kde=True, palette={'falsa': 'red', 'verdadeira': 'blue'})
plt.title('Distribuição de Comprimento do Corpo da Notícia (Verdadeiras vs Falsas)')
plt.xlabel('Número de Palavras no Texto')
plt.ylabel('Frequência (Qtd de Notícias)')
# Limitando o eixo X para tirar os outliers extremos e ver o grosso dos dados (ex: até 1500 palavras)
plt.xlim(0, 1500)
plt.savefig(os.path.join(output_dir, 'distribuicao_comprimento.png'))
plt.close()

# 5. Plotar Análise de Sensacionalismo (Exclamações no Título)
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x='classe', y='qtd_exclamacoes', palette={'falsa': 'red', 'verdadeira': 'blue'})
plt.title('Uso de Exclamações (!) no Título - Proxy de Sensacionalismo')
plt.ylabel('Média de Exclamações por Notícia')
plt.savefig(os.path.join(output_dir, 'analise_sensacionalismo.png'))
plt.close()

print("\n✅ Análise concluída com sucesso!")
print(f"📈 Gráficos gerados e salvos na pasta: {output_dir}")

print("\n--- RESUMO ESTATÍSTICO ---")
medias = df.groupby('classe')[['len_titulo', 'len_subtitulo', 'len_texto_completo', 'qtd_exclamacoes']].mean()
print(medias)
