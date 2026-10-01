"""
=================================================================
FASE 4.6: TREINAMENTO V4 COM CHUNKING (FATIAMENTO E BALANCEAMENTO)
Destaques:
  1. Dataset v4: notícias verdadeiras fatiadas para equiparar
     o tamanho com as falsas (média ~50 palavras cada).
  2. MAX_LEN=128: ideal para textos fatiados, 4x mais rápido que 512.
  3. Fim do atalho do comprimento: o BERT é forçado a ler semântica.
  4. Rastreamento completo no MLflow (experimento: FakeNews_Bertimbau_V4_Chunking).
=================================================================
"""
import pandas as pd
import numpy as np
import mlflow
import os
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

print("🚀 Iniciando Treinamento V4 (Dataset Chunked / Sem Viés de Tamanho)")

# 1. Configuração do MLflow
mlflow.set_experiment("FakeNews_Bertimbau_V4_Chunking")

# Parâmetros V4
MODEL_NAME = 'neuralmind/bert-base-portuguese-cased'
N_ITERATIONS = 3
EPOCHS = 3
BATCH_SIZE = 16       # Com MAX_LEN=128 podemos usar batch 16 confortavelmente
GRAD_ACCUM = 2       # Effective batch size = 32
MAX_LEN = 128        # Cobre 98%+ dos chunks de ~50 palavras com folga
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
DATA_PATH = 'data/processed/dataset_treino_chunking_v4.csv'

# 2. Carregar Dataset Versionado V4
print(f"📖 Lendo dados de {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=['texto', 'label'])
df['label'] = df['label'].astype(int)
print(f"Dataset V4 carregado com {len(df)} amostras balanceadas.")

df['num_tokens_aprox'] = df['texto'].str.split().str.len()
print(f"\n📊 Estatísticas de comprimento no V4:")
print(f"  Falsas:      média={df[df['label']==0]['num_tokens_aprox'].mean():.1f} palavras")
print(f"  Verdadeiras: média={df[df['label']==1]['num_tokens_aprox'].mean():.1f} palavras")
print(f"  Ratio:       {df[df['label']==1]['num_tokens_aprox'].mean() / df[df['label']==0]['num_tokens_aprox'].mean():.2f}x (Equilíbrio Semântico!)")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(examples['texto'], padding="max_length", truncation=True, max_length=MAX_LEN)

f1_scores = []

for i in range(N_ITERATIONS):
    print(f"\n{'='*55}")
    print(f"🔥 Rodada {i+1} de {N_ITERATIONS} (Seed: {42+i}) | MAX_LEN={MAX_LEN}")
    print(f"{'='*55}")
    
    with mlflow.start_run(run_name=f"V4_Chunking_Fold_{i+1}"):
        mlflow.log_param("fold_index", i+1)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("effective_batch_size", BATCH_SIZE * GRAD_ACCUM)
        mlflow.log_param("max_len", MAX_LEN)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("weight_decay", WEIGHT_DECAY)
        mlflow.log_param("random_seed", 42+i)
        mlflow.log_param("version", "V4_Chunking_Balanced")
        
        # Split estratificado 70/30
        X_train, X_test, y_train, y_test = train_test_split(
            df['texto'].tolist(), df['label'].tolist(), 
            test_size=0.3, random_state=42+i, stratify=df['label'].tolist()
        )
        
        train_dataset = Dataset.from_dict({'texto': X_train, 'label': y_train})
        test_dataset = Dataset.from_dict({'texto': X_test, 'label': y_test})
        
        print(f"Tokenizando ({len(train_dataset)} treino / {len(test_dataset)} teste)...")
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        test_dataset = test_dataset.map(tokenize_function, batched=True)
        
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=2,
            id2label={0: 'falsa', 1: 'verdadeira'},
            label2id={'falsa': 0, 'verdadeira': 1}
        )
        
        output_dir = f'./logs/results_v4_fold_{i}'
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            report_to="none",
            fp16=False,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics
        )
        
        print("Treinando V4 com Chunking...")
        trainer.train()
        
        print("Avaliando...")
        metrics = trainer.evaluate()
        
        mlflow.log_metric("eval_accuracy", metrics['eval_accuracy'])
        mlflow.log_metric("eval_precision", metrics['eval_precision'])
        mlflow.log_metric("eval_recall", metrics['eval_recall'])
        mlflow.log_metric("eval_f1", metrics['eval_f1'])
        
        f1_scores.append(metrics['eval_f1'])
        print(f"✅ Rodada {i+1} concluída. F1-Score: {metrics['eval_f1']:.4f}")

media_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)

print(f"\n🏆 RESULTADO FINAL (V4 CHUNKING) 🏆")
print(f"Média do F1-Score em {N_ITERATIONS} rodadas: {media_f1:.4f} ± {std_f1:.4f}")
print("Para ver o painel do MLflow, rode no terminal: mlflow ui")
