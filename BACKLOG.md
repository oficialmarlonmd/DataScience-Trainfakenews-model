# 📋 Backlog do Projeto: Fake News Classification (Bertimbau)

Este documento centraliza as tarefas (tasks) acordadas para a evolução do modelo de Machine Learning, garantindo o histórico de decisões técnicas e o fluxo de trabalho profissional de Ciência de Dados.

---

## 🟢 Fase 1: Governança e Versionamento (Em Andamento)
- [x] **Task 1.1:** Limpar arquivos residuais e organizar estrutura de pastas (`data/`, `src/`, `logs/`).
- [x] **Task 1.2:** Inicializar Git e criar `.gitignore` para bloquear arquivos pesados (.pkl, checkpoints).
- [x] **Task 1.3:** Criar branches estratégicas (`master` para produção, `development` para testes).
- [x] **Task 1.4:** Realizar o `git push` inicial para o GitHub remoto (Usuário autenticando).

---

## 🟡 Fase 2: Análise Exploratória de Dados (EDA) no Dataset Expandido
*Foco: Entender as features do novo dataset `noticias_balanceadas_expandidas_limpas.csv` antes do treinamento.*
- [x] **Task 2.1:** Script de EDA para plotar histogramas de **Comprimento de Texto (Tokens)** das notícias (Verdadeiras vs Falsas).
- [x] **Task 2.2:** Script de **Análise de Sentimento**. Escanear o texto e criar métricas de polaridade (-1 a +1) para provar a hipótese de que fakes são emocionalmente extremas.
- [x] **Task 2.3:** Documentar os achados estatísticos (os gráficos) para basear nossa decisão de poda/tamanho máximo (max_len).

---

## 🟠 Fase 3: Engenharia de Features (Data Prep)
*Foco: Otimizar como o modelo consome a informação jornalística.*
- [x] **Task 3.1:** Desenvolver o script de concatenação inteligente de colunas. Ao invés de usar apenas o texto, vamos unir `[CLS] Titulo [SEP] Subtitulo [SEP] Texto [SEP]`.
- [x] **Task 3.2:** Normalizar o dataset (lidar com Nulos em subtítulos, balancear as classes se necessário).
- [x] **Task 3.3:** Salvar a nova versão do dataset como `dataset_treino_multifeature_v2.csv`.

---

## 🔴 Fase 4: Treinamento Robusto (Monte Carlo + MLOps)
*Foco: Elevar o rigor matemático e implantar rastreamento profissional de experimentos.*
- [x] **Task 4.1:** Instalar e configurar **MLflow** para rastreamento de experimentos (Feast removido por não ser o mais adequado para features textuais simples).
- [x] **Task 4.2:** Criar script `train_monte_carlo.py` integrado com MLflow (log de parâmetros, precisão, recall, f1, acurácia).
- [x] **Task 4.3:** Executar o loop de Monte Carlo Cross-Validation (70/30) N vezes, rastreando cada rodada (run) no MLflow.
- [x] **Task 4.4:** Investigação forense do modelo V2 e diagnóstico de viés de comprimento (Shortcut Learning).
- [x] **Task 4.5:** Retreino recalibrado V3 (MAX_LEN=512, Gradient Accumulation).
- [x] **Task 4.6:** Engenharia de Features V4: Fatiamento (Chunking) das notícias verdadeiras para igualar o tamanho (~50 palavras) e gerar `dataset_treino_chunking_v4.csv` balanceado 1:1.

---

## 🟣 Fase 5: MLOps Model Serving e Entrega
*Foco: Empacotamento de alto nível para consumo via API.*
- [x] **Task 5.1:** Escolher o modelo campeão pelo painel do MLflow.
- [x] **Task 5.2:** Empacotar o modelo utilizando **BentoML** (ao invés de apenas um Pickle simples) para criar um serviço pronto para produção.
- [x] **Task 5.3:** Atualizar o Relatório/Dossiê Técnico com a arquitetura de MLOps e testes formais.
- [x] **Task 5.4:** Commitar e mergear a branch `development` na `master` com a tag `v2.0-mlops-ready`.
