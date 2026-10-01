import os
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore')

# 1. Hardware Detection (Apple Silicon MPS / CUDA / CPU)
def get_device():
    if torch.cuda.is_available():
        device = "cuda"
        print("🚀 GPU CUDA ativada!")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("🍏 Aceleração Apple Silicon (MPS) ativada! Pronto para rodar no seu Mac M4.")
    else:
        device = "cpu"
        print("⚠️ Usando CPU. O treinamento será mais lento.")
    return device

device = get_device()

# 2. Carregar e Preparar o Novo Dataset Normalizado
csv_path = "dataset_final_treinamento.csv"
print(f"\n📂 Carregando dataset normalizado: {csv_path}...")
df = pd.read_csv(csv_path)

# Filtrar e tratar valores nulos
df['texto_limpo'] = df['texto_limpo'].fillna('').astype(str).str.strip()
df = df[df['texto_limpo'].str.len() > 5].copy()

# Garantir classes esperadas e mapeamento
df = df[df['classe'].isin(['falsa', 'verdadeira'])].copy()
if 'label' not in df.columns:
    df['label'] = df['classe'].map({'falsa': 0, 'verdadeira': 1})

print("\nDistribuição das Classes no Dataset Normalizado:")
print(df['classe'].value_counts())
print(f"Total de exemplos: {len(df)}")

# Divisão Treino e Teste (Estratificada)
X = df['texto_limpo'].tolist()
y = df['label'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nConjunto de Treino: {len(X_train)} amostras")
print(f"Conjunto de Teste: {len(X_test)} amostras")

# 3. Carregar o Tokenizador do BERTimbau
model_name = "neuralmind/bert-base-portuguese-cased"
print(f"\nCarregando tokenizador: {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 4. Classe do Dataset para o PyTorch
class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

print("\nPreparando PyTorch Datasets...")
train_dataset = FakeNewsDataset(X_train, y_train, tokenizer, max_len=128)
test_dataset = FakeNewsDataset(X_test, y_test, tokenizer, max_len=128)
print("✓ Datasets prontos!")

# 5. Carregar Modelo Pré-treinado
print(f"\nCarregando modelo {model_name}...")
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=2,
    id2label={0: 'falsa', 1: 'verdadeira'},
    label2id={'falsa': 0, 'verdadeira': 1}
)

# 6. Métricas de Avaliação
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# 7. Configurações de Treinamento
output_dir = './bert_fakenews_results'
final_model_dir = './bert_fakenews_model'

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=2,                 # 2 épocas para excelente convergência
    per_device_train_batch_size=32,     # Batch de 32 aproveitando os 24GB do M4
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    warmup_steps=50,
    weight_decay=0.01,
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none"
)

# 8. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# 9. Treinamento
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 INICIANDO TREINAMENTO COM BERTIMBAU NO MAC M4")
    print("="*50 + "\n")
    
    train_result = trainer.train()
    print("\n✓ Treinamento concluído com sucesso!")
    
    # Avaliação no conjunto de teste
    print("\n" + "="*50)
    print("📊 AVALIAÇÃO FINAL NO CONJUNTO DE TESTE")
    print("="*50)
    eval_metrics = trainer.evaluate()
    for k, v in eval_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    # Previsões detalhadas no conjunto de teste
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    
    print("\n📋 Relatório de Classificação Detalhado:")
    print(classification_report(y_test, preds, target_names=['Falsa', 'Verdadeira'], digits=4))
    
    print("\nMatriz de Confusão:")
    print(confusion_matrix(y_test, preds))
    
    # 10. Salvar Modelo e Tokenizador para uso futuro
    print(f"\n💾 Salvando modelo e tokenizador em '{final_model_dir}'...")
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    print("✓ Modelo salvo e pronto para inferência!")
