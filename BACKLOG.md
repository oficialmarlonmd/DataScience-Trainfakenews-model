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

## 🔴 Fase 4: Treinamento Robusto e Validação Estatística
*Foco: Elevar o rigor matemático do treinamento usando Validação Cruzada.*
- [ ] **Task 4.1:** Criar script `train_monte_carlo.py` focado em **Monte Carlo Cross-Validation** (Embaralhar e treinar repetidas vezes na proporção 70/30).
- [ ] **Task 4.2:** Incorporar logging de resultados (salvar as métricas precisão, recall, f1, acurácia de cada rodada em um `.json` automático).
- [ ] **Task 4.3:** Calcular a Média e o Desvio Padrão das métricas finais para provar que o modelo é inabalável.

---

## 🟣 Fase 5: Entrega e Versionamento do Novo Modelo
- [ ] **Task 5.1:** Escolher o modelo campeão das rodadas de validação cruzada.
- [ ] **Task 5.2:** Exportar o modelo campeão (pipeline completo via Pickle / Save_Pretrained).
- [ ] **Task 5.3:** Atualizar o Relatório/Dossiê Técnico com a nova arquitetura e resultados.
- [ ] **Task 5.4:** Commitar e mergear a branch `development` na `master` com a tag `v2.0-modelo-multifeature`.
