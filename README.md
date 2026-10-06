# 📰 Fake News Classification Model (BERTimbau)

Este repositório contém todo o pipeline de Machine Learning e MLOps desenvolvido para a classificação e detecção de Fake News em textos de língua portuguesa, utilizando o modelo **BERTimbau** (`neuralmind/bert-base-portuguese-cased`).

O projeto abrange todo o ciclo de vida de Ciência de Dados: análise exploratória (EDA), engenharia de features jornalísticas, investigação forense de atalhos de aprendizado (*shortcut learning*), retreino com técnica de *chunking* desenviesado e empacotamento com **BentoML**.

---

## 🏆 Modelo Campeão em Produção (V4 — Chunking)

O modelo oficial de produção é a **Versão 4 (V4)**, que superou o viés de tamanho das notícias através de fatiamento semântico (*chunking* 1:1), atingindo índices de excelência no teste cego com 100 notícias reais (50 Falsas e 50 Verdadeiras):

| Métrica | Resultado V4 | Status |
| :--- | :---: | :---: |
| **Acurácia Global** | **98.00%** (98 / 100) | 🏆 Campeão Absoluto |
| **Recall em Falsas (Sensibilidade)** | **100.00%** (50 / 50) | 🎯 Zero Notícias Falsas Deixadas Passar |
| **Precisão em Falsas** | **96.15%** | Alta confiabilidade no alerta |
| **Precisão em Verdadeiras** | **100.00%** | Todo conteúdo rotulado como real é legítimo |
| **Recall em Verdadeiras** | **96.00%** (48 / 50) | Apenas 2 falsos positivos conservadores |
| **Latência por Notícia** | **~14 ms** | Acelerado via Apple Silicon Metal (MPS) |

---

## 🔬 Evolução Científica das Versões (`src/`)

Para garantir a rastreabilidade e governança de Machine Learning, o repositório mantém a árvore completa de evolução experimental:

- **`src/v1_baseline/`**: Modelo inicial com texto bruto (Baseline 92% no dataset limpo original).
- **`src/v2_multifeature/`**: Concatenação inteligente de atributos `[CLS] Titulo [SEP] Subtitulo [SEP] Texto`. Diagnosticado com *Shortcut Learning* (acurácia caiu para 41.7% no teste cego porque notícias verdadeiras eram mais longas que as falsas).
- **`src/v3_recalibrado/`**: Aumento de contexto para `MAX_LEN=512`. Atingiu 84.00% de acurácia, mas demandou 16h+ de treino e recall de 68% em notícias falsas.
- **`src/v4_chunking/` (Campeão)**: Fatiamento de notícias verdadeiras em blocos médios de 35 palavras, equalizando a distribuição com as notícias falsas (razão de tamanho 0.97x). Treinado em ~25 minutos com `MAX_LEN=128`, alcançando **98.00% de acurácia**.

---

## 📂 Estrutura do Repositório

```text
fake_news/
├── data/
│   └── processed/
│       ├── dataset_final_treinamento.csv          # Dataset V1
│       ├── dataset_treino_multifeature_v2.csv      # Dataset V2
│       ├── dataset_treino_chunking_v4.csv          # Dataset V4 (Chunking)
│       └── validacao_100_multifeature.csv          # Benchmark Oficial (50 Falsas / 50 Verdadeiras)
├── src/
│   ├── v1_baseline/                               # Códigos V1 (Treino, validação e pickle legado)
│   ├── v2_multifeature/                           # Códigos V2 (EDA, Monte Carlo e forense)
│   ├── v3_recalibrado/                            # Códigos V3 (MAX_LEN=512, export_bentoml_v3.py)
│   ├── v4_chunking/                               # Códigos V4 (Chunking, treino e export_bentoml_v4.py)
│   └── pipeline/                                  # Scripts auxiliares de pipeline e validação final
├── tests/
│   └── test_model_v4_accuracy.py                  # Suíte formal de testes unitários automatizados
├── BACKLOG.md                                     # Controle de fases e governança técnica
└── README.md                                      # Documentação principal
```

---

## 🧪 Testes Automatizados

A suíte formal de testes unitários valida a acurácia, balanceamento e extinção de viés sobre as 100 notícias de teste:

```bash
python3 -m unittest tests/test_model_v4_accuracy.py -v
```

Saída esperada:
```text
test_01_integridade_dataset_50_50 ... ok
test_02_acuracia_50_falsas (50/50 - 100.00%) ... ok
test_03_acuracia_50_verdadeiras (48/50 - 96.00%) ... ok
test_04_acuracia_global (98/100 - 98.00%) ... ok
test_05_desenviesamento_e_calibracao ... ok
```

---

## 📦 Model Serving com BentoML

Todos os modelos estão registrados no BentoML Model Store para consumo por back-ends e APIs:

```bash
python3 -m bentoml models list
```

```text
Tag                                Module           Size        Creation Time       
fakenews_bert_v4:afob72gbq26xt6su  bentoml.pytorch  415.63 MiB  2026-10-06 10:01:11 
fakenews_bert_v3:5vcrv6gbq6deh6su  bentoml.pytorch  415.63 MiB  2026-10-06 10:14:57 
fakenews_bert_v2:d5mnksf5uwjqd6su  bentoml.pytorch  415.63 MiB  2026-10-01 11:33:51 
```

Para re-exportar ou atualizar o modelo V4 no BentoML:
```bash
python3 src/v4_chunking/export_bentoml_v4.py
```

---

## 🌿 Governança de Branches
- **`master`**: Branch oficial de produção contendo a versão estável e o modelo campeão V4 (`v2.0-mlops-ready`).
- **`development`**: Branch de integração contínua e histórico de desenvolvimento.
- **`feature/phase*`**: Branches de ciclo de vida das fases 1 a 5.
