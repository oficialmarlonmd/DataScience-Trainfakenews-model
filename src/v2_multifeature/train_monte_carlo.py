import pandas as pd
import numpy as np
import mlflow
import os
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# ==========================================
# FASE 4: TREINAMENTO ROBUSTO COM MLOPS
# ==========================================

print("🚀 Iniciando MLOps Pipeline: Monte Carlo Cross-Validation com MLflow")

# Configuração do MLflow
mlflow.set_experiment("FakeNews_Bertimbau_V2")

# Parâmetros Globais
MODEL_NAME = 'neuralmind/bert-base-portuguese-cased'
N_ITERATIONS = 3 # Número de Folds (Monte Carlo)
EPOCHS = 2
BATCH_SIZE = 32
MAX_LEN = 128
DATA_PATH = 'data/processed/dataset_treino_multifeature_v2.csv'

# 1. Carregar Dataset
print(f"Lendo dados de {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=['texto', 'label'])
# Garantir que a label seja inteira (0 ou 1)
df['label'] = df['label'].map({'falsa': 0, 'verdadeira': 1, 0: 0, 1: 1, '0': 0, '1': 1})
df['label'] = df['label'].astype(int)
print(f"Dataset carregado com {len(df)} amostras.")

# Função de métricas
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

# Tokenizador
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(examples['texto'], padding="max_length", truncation=True, max_length=MAX_LEN)

# Loop de Monte Carlo
f1_scores = []

for i in range(N_ITERATIONS):
    print(f"\n{'='*40}")
    print(f"🔥 Iniciando Rodada {i+1} de {N_ITERATIONS} (Seed: {42+i})")
    print(f"{'='*40}")
    
    # Criar uma nova Run no MLflow para cada rodada
    with mlflow.start_run(run_name=f"Monte_Carlo_Fold_{i+1}"):
        
        # Log de parâmetros
        mlflow.log_param("fold_index", i+1)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("max_len", MAX_LEN)
        mlflow.log_param("random_seed", 42+i)
        
        # Shuffle Split 70/30
        X_train, X_test, y_train, y_test = train_test_split(
            df['texto'].tolist(), df['label'].tolist(), 
            test_size=0.3, random_state=42+i, stratify=df['label'].tolist()
        )
        
        # Converter para formato Dataset HuggingFace
        train_dataset = Dataset.from_dict({'texto': X_train, 'label': y_train})
        test_dataset = Dataset.from_dict({'texto': X_test, 'label': y_test})
        
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        test_dataset = test_dataset.map(tokenize_function, batched=True)
        
        # Carregar modelo zerado a cada rodada para não herdar pesos anteriores
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=2,
            id2label={0: 'falsa', 1: 'verdadeira'},
            label2id={'falsa': 0, 'verdadeira': 1}
        )
        
        training_args = TrainingArguments(
            output_dir=f'./logs/results_fold_{i}',
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            learning_rate=2e-5,
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            report_to="none" # Desliga W&B para usar só MLflow
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics
        )
        
        # Treinar
        print("Treinando...")
        trainer.train()
        
        # Avaliar
        print("Avaliando...")
        metrics = trainer.evaluate()
        
        # Log de métricas no MLflow
        mlflow.log_metric("eval_accuracy", metrics['eval_accuracy'])
        mlflow.log_metric("eval_precision", metrics['eval_precision'])
        mlflow.log_metric("eval_recall", metrics['eval_recall'])
        mlflow.log_metric("eval_f1", metrics['eval_f1'])
        
        f1_scores.append(metrics['eval_f1'])
        
        print(f"✅ Rodada {i+1} concluída. F1-Score: {metrics['eval_f1']:.4f}")

# Calcular e mostrar média final
media_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)

print("\n🏆 RESULTADO FINAL (MONTE CARLO CROSS-VALIDATION) 🏆")
print(f"Média do F1-Score em {N_ITERATIONS} rodadas: {media_f1:.4f} ± {std_f1:.4f}")
print("Para ver o painel do MLflow, rode no terminal: mlflow ui")
