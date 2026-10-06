# 🗂️ Arquitetura e Organização do Código (`src/`)

Este diretório está organizado em módulos versionados, permitindo rastrear a evolução científica de cada modelo de Machine Learning, seus datasets e scripts de validação:

```
src/
├── v1_baseline/         # Modelo inicial BERTimbau (Baseline com textos crus)
│   ├── analise_exploratoria.py
│   ├── preparar_dataset_final.py
│   ├── train_bertimbau.py
│   ├── validar_modelo.py
│   ├── test_predict.py
│   └── exportar_pickle.py
│
├── v2_multifeature/     # Features compostas [CLS] Titulo [SEP] Subtitulo [SEP] Texto
│   ├── preparar_dataset_multifeature.py
│   ├── train_monte_carlo.py
│   ├── export_bentoml.py
│   ├── validar_v2.py
│   ├── validar_multifeature.py
│   └── investigar_modelo.py
│
├── v3_recalibrado/      # Hipótese de aumento de MAX_LEN=512 (Staging pós-forense)
│   ├── gerar_validacao_100.py
│   ├── train_v3_recalibrado.py
│   ├── validar_v3.py
│   └── export_bentoml_v3.py
│
├── v4_chunking/         # Modelo Campeão: Fatiamento das notícias longas (Desenviesamento 1:1)
│   ├── preparar_dataset_chunking_v4.py
│   ├── train_v4_chunking.py
│   ├── validar_v4.py
│   └── export_bentoml_v4.py
│
└── pipeline/            # Orquestração e automação de experimentos
    ├── monitorar_v3_e_executar_v4.py
    └── validar_final.py
```

### 📋 Guia Rápido de Execução

- **Treinar o modelo atual (V4 - Chunking)**:
  ```bash
  python3 src/v4_chunking/train_v4_chunking.py
  ```
- **Validar o modelo V4 nas notícias reais**:
  ```bash
  python3 src/v4_chunking/validar_v4.py
  ```
- **Monitorar o término do V3 e engatar o V4 automaticamente**:
  ```bash
  python3 src/pipeline/monitorar_v3_e_executar_v4.py
  ```
