"""
=================================================================
FASE 4.5: RETREINO RECALIBRADO (V3)
Correções aplicadas após investigação forense:
  1. MAX_LEN aumentado de 128 → 512 (BERT lê o texto completo)
  2. Gradient Accumulation para compensar a memória com MAX_LEN maior
  3. Weight Decay para regularização
  4. Learning Rate menor para fine-tuning mais fino
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

print("🚀 Iniciando Retreino Recalibrado V3 (Pós-Investigação Forense)")

# Configuração do MLflow
mlflow.set_experiment("FakeNews_Bertimbau_V3_Recalibrado")

# Parâmetros RECALIBRADOS
MODEL_NAME = 'neuralmind/bert-base-portuguese-cased'
N_ITERATIONS = 3
EPOCHS = 3          # +1 epoch para compensar o learning rate menor
BATCH_SIZE = 8      # Reduzido de 32 para 8 (MAX_LEN maior consome mais memória)
GRAD_ACCUM = 4      # Gradient Accumulation: 8 * 4 = batch efetivo de 32
MAX_LEN = 512       # CORREÇÃO PRINCIPAL: de 128 → 512
LEARNING_RATE = 1e-5  # Mais conservador que o 2e-5 anterior
WEIGHT_DECAY = 0.01   # Regularização para evitar overfitting
DATA_PATH = 'data/processed/dataset_treino_multifeature_v2.csv'

# 1. Carregar Dataset
print(f"Lendo dados de {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=['texto', 'label'])
df['label'] = df['label'].map({'falsa': 0, 'verdadeira': 1, 0: 0, 1: 1, '0': 0, '1': 1})
df['label'] = df['label'].astype(int)
print(f"Dataset carregado com {len(df)} amostras.")

# Estatísticas de comprimento
df['num_tokens_aprox'] = df['texto'].str.split().str.len()
print(f"\n📊 Estatísticas de comprimento:")
print(f"  Falsas:      média={df[df['label']==0]['num_tokens_aprox'].mean():.0f} palavras")
print(f"  Verdadeiras: média={df[df['label']==1]['num_tokens_aprox'].mean():.0f} palavras")
print(f"  Com MAX_LEN=512, {(df['num_tokens_aprox'] <= 512).mean()*100:.1f}% das amostras serão lidas por completo")

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
    print(f"\n{'='*50}")
    print(f"🔥 Rodada {i+1} de {N_ITERATIONS} (Seed: {42+i}) | MAX_LEN={MAX_LEN}")
    print(f"{'='*50}")
    
    with mlflow.start_run(run_name=f"V3_Recalibrado_Fold_{i+1}"):
        
        # Log de parâmetros (incluindo os novos)
        mlflow.log_param("fold_index", i+1)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("effective_batch_size", BATCH_SIZE * GRAD_ACCUM)
        mlflow.log_param("max_len", MAX_LEN)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("weight_decay", WEIGHT_DECAY)
        mlflow.log_param("random_seed", 42+i)
        mlflow.log_param("version", "V3_Recalibrado")
        
        # Shuffle Split 70/30
        X_train, X_test, y_train, y_test = train_test_split(
            df['texto'].tolist(), df['label'].tolist(), 
            test_size=0.3, random_state=42+i, stratify=df['label'].tolist()
        )
        
        train_dataset = Dataset.from_dict({'texto': X_train, 'label': y_train})
        test_dataset = Dataset.from_dict({'texto': X_test, 'label': y_test})
        
        print("Tokenizando (MAX_LEN=512, pode demorar um pouco mais)...")
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        test_dataset = test_dataset.map(tokenize_function, batched=True)
        
        # Carregar modelo zerado
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=2,
            id2label={0: 'falsa', 1: 'verdadeira'},
            label2id={'falsa': 0, 'verdadeira': 1}
        )
        
        training_args = TrainingArguments(
            output_dir=f'./logs/results_v3_fold_{i}',
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
            fp16=False,  # MPS não suporta FP16
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics
        )
        
        print("Treinando V3...")
        trainer.train()
        
        print("Avaliando...")
        metrics = trainer.evaluate()
        
        mlflow.log_metric("eval_accuracy", metrics['eval_accuracy'])
        mlflow.log_metric("eval_precision", metrics['eval_precision'])
        mlflow.log_metric("eval_recall", metrics['eval_recall'])
        mlflow.log_metric("eval_f1", metrics['eval_f1'])
        
        f1_scores.append(metrics['eval_f1'])
        
        print(f"✅ Rodada {i+1} concluída. F1-Score: {metrics['eval_f1']:.4f}")

# Resultado final
media_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)

print(f"\n🏆 RESULTADO FINAL (V3 RECALIBRADO) 🏆")
print(f"Média do F1-Score em {N_ITERATIONS} rodadas: {media_f1:.4f} ± {std_f1:.4f}")
print(f"Comparar com V2: 0.8930 ± 0.0017")
print("Para ver o painel do MLflow, rode no terminal: mlflow ui")
