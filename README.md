# 📰 Fake News Classification Model (Bertimbau)

Este repositório contém todo o pipeline de Machine Learning desenvolvido para a classificação e detecção de Fake News em textos de língua portuguesa, utilizando o modelo **Bertimbau** (BERT adaptado para o português brasileiro).

O foco do projeto é construir um fluxo robusto de Data Science e MLOps, abrangendo desde a análise exploratória e preparação de dados até o treinamento com validação rigorosa e a conteinerização do modelo para serving em produção.

---

## 🚀 Funcionalidades e Pipeline de ML

O desenvolvimento deste modelo seguiu fases rigorosas de Ciência de Dados:

### 1. Análise Exploratória de Dados (EDA)
- Análise aprofundada de distribuição do comprimento de tokens nas notícias.
- Análise de sentimento para capturar extremismos emocionais, com a premissa de que *fake news* possuem forte carga de polaridade.

### 2. Engenharia de Features
- Concatenação estruturada de atributos jornalísticos utilizando tokens especiais do BERT: `[CLS] Titulo [SEP] Subtitulo [SEP] Texto [SEP]`.
- Limpeza e balanceamento de classes visando a robustez do modelo em cenários reais.

### 3. Treinamento Robusto e Rastreamento (MLOps)
- Utilização de **Monte Carlo Cross-Validation** (70/30) para atestar a capacidade de generalização do modelo em múltiplas rodadas.
- Rastreamento completo de parâmetros, métricas (Precision, Recall, F1-Score, Accuracy) e artefatos de modelo através do **MLflow**.

### 4. Model Serving & Deployment
- O modelo campeão é empacotado e preparado para servir inferências (API) através da integração com o framework **BentoML**.

---

## 📂 Estrutura do Repositório

- `src/`: Scripts principais em Python responsáveis pelo core do ML.
  - `analise_exploratoria.py`: Script de EDA e análise de sentimentos.
  - `preparar_dataset_final.py` e `preparar_dataset_multifeature.py`: Scripts de processamento, normalização e engenharia de features do texto.
  - `train_bertimbau.py` e `train_monte_carlo.py`: Rotinas de treinamento do classificador utilizando Bertimbau com integração ao MLflow.
  - `validar_modelo.py` e `test_predict.py`: Scripts focados em testar o modelo treinado.
  - `export_bentoml.py`: Preparação e empacotamento do modelo usando BentoML.
  - `exportar_pickle.py`: Scripts legados de exportação do modelo em formato `.pkl`.
- `data/`: Diretório destinado ao armazenamento de datasets originais e processados (geralmente ignorados no versionamento se forem muito grandes).
- `logs/`: Saídas geradas durante o processamento e o treinamento.
- `mlflow.db`: Banco de dados contendo o registro histórico dos experimentos do MLflow.
- `notebooks/`: Notebooks Jupyter utilizados para prototipação e testes rápidos.
- `BACKLOG.md`: Histórico de evolução e controle de atividades das etapas de vida do modelo.

---

## 🛠️ Tecnologias Utilizadas
- **Linguagem**: Python
- **Modelagem NLP**: HuggingFace Transformers, Bertimbau
- **MLOps e Tracking**: MLflow
- **Model Serving**: BentoML
- **Análise e Manipulação**: Pandas, Scikit-learn, Matplotlib, Seaborn

---

## ⚙️ Como Utilizar o Repositório

### Instalação de Dependências
Recomenda-se o uso de um ambiente virtual (ex: `venv` ou `conda`) para instalar as dependências do projeto.
```bash
pip install -r requirements.txt
```

### Visualizando Experimentos no MLflow
Para conferir o histórico dos treinamentos (Métricas e Hiperparâmetros):
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### Inferência Local / Deploy BentoML
Para subir o servidor do modelo com o BentoML:
```bash
bentoml serve src.export_bentoml:svc --reload
```
*(Confirme no código o nome do serviço (svc) que está sendo exportado antes de servir)*.

---

*Nota: A branch principal deste projeto é a `master`.*
