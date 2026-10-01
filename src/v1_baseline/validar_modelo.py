import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class TextBatchDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        encoding = self.tokenizer(
            str(self.texts[item]),
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

def validar_modelo(csv_path="validacao_inedita.csv", model_dir="./bert_fakenews_model"):
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "="*65)
    print("🔬 VALIDAÇÃO DO MODELO COM NOTÍCIAS INÉDITAS")
    print(f"📂 Dataset: {csv_path}")
    print(f"🧠 Modelo: {model_dir} | Dispositivo: {device.upper()}")
    print("="*65)

    # Carregar dados
    df = pd.read_csv(csv_path)
    texts = df['texto'].astype(str).tolist()
    labels_reais = df['classe_real'].map({'falsa': 0, 'verdadeira': 1}).values

    print(f"\nTotal de notícias para validação: {len(texts)}")
    print(f"  • Verdadeiras: {(labels_reais == 1).sum()}")
    print(f"  • Falsas: {(labels_reais == 0).sum()}")

    # Carregar modelo
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    dataset = TextBatchDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    all_preds = []
    all_probs = []

    print("\n⚡ Processando previsões em lote...")
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)
            all_preds.extend(preds)
            all_probs.extend(probs)

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    classes = ['FALSA', 'VERDADEIRA']

    # === MÉTRICAS GERAIS ===
    acc = accuracy_score(labels_reais, all_preds)
    print("\n" + "="*65)
    print(f"🎯 ACURÁCIA GERAL: {acc*100:.2f}% ({int(acc * len(texts))}/{len(texts)} acertos)")
    print("="*65)

    print("\n📋 RELATÓRIO DE CLASSIFICAÇÃO DETALHADO:")
    print(classification_report(labels_reais, all_preds, target_names=['FALSA', 'VERDADEIRA'], digits=4))

    cm = confusion_matrix(labels_reais, all_preds)
    print("📊 MATRIZ DE CONFUSÃO:")
    print(f"                 Previsto FALSA   Previsto VERDADEIRA")
    print(f"  Real FALSA       {cm[0][0]:>5}             {cm[0][1]:>5}")
    print(f"  Real VERDADEIRA  {cm[1][0]:>5}             {cm[1][1]:>5}")

    # === DETALHAMENTO POR CLASSE ===
    n_falsas = (labels_reais == 0).sum()
    n_verdadeiras = (labels_reais == 1).sum()
    acertos_falsas = ((labels_reais == 0) & (all_preds == 0)).sum()
    acertos_verdadeiras = ((labels_reais == 1) & (all_preds == 1)).sum()

    print(f"\n{'='*65}")
    print(f"📰 RESULTADO POR CLASSE:")
    print(f"{'='*65}")
    print(f"  ❌ FALSAS:      {acertos_falsas}/{n_falsas} acertos ({acertos_falsas/n_falsas*100:.1f}%)")
    print(f"  ✅ VERDADEIRAS: {acertos_verdadeiras}/{n_verdadeiras} acertos ({acertos_verdadeiras/n_verdadeiras*100:.1f}%)")

    # === DETALHES DE CADA PREVISÃO ===
    print(f"\n{'='*65}")
    print("🔎 DETALHES INDIVIDUAIS DE TODAS AS PREVISÕES:")
    print(f"{'='*65}\n")

    erros = []
    for i in range(len(texts)):
        real = classes[labels_reais[i]]
        pred = classes[all_preds[i]]
        conf = all_probs[i][all_preds[i]] * 100
        status = "✅" if labels_reais[i] == all_preds[i] else "❌ ERRO"
        txt = texts[i][:80] + "..." if len(texts[i]) > 80 else texts[i]

        if labels_reais[i] != all_preds[i]:
            erros.append((i+1, txt, real, pred, conf))
            print(f"  {i+1:>2}. {status} | Real: {real:<12} | Pred: {pred:<12} | Conf: {conf:>5.1f}% | {txt}")
        else:
            print(f"  {i+1:>2}. {status}    | Real: {real:<12} | Pred: {pred:<12} | Conf: {conf:>5.1f}% | {txt}")

    if erros:
        print(f"\n{'='*65}")
        print(f"⚠️  LISTA DE ERROS ({len(erros)} de {len(texts)}):")
        print(f"{'='*65}")
        for num, txt, real, pred, conf in erros:
            print(f"  #{num}: Notícia era {real} mas o modelo previu {pred} ({conf:.1f}%)")
            print(f"        \"{txt}\"\n")

    # Salvar resultados
    resultado_df = pd.DataFrame({
        'noticia': texts,
        'classe_real': [classes[l] for l in labels_reais],
        'predicao': [classes[p] for p in all_preds],
        'confianca_falsa_%': [round(p[0]*100, 2) for p in all_probs],
        'confianca_verdadeira_%': [round(p[1]*100, 2) for p in all_probs],
        'acertou': ['SIM' if labels_reais[i] == all_preds[i] else 'NÃO' for i in range(len(texts))]
    })
    resultado_df.to_csv('resultado_validacao.csv', index=False)
    print(f"\n💾 Resultados completos salvos em 'resultado_validacao.csv'")

if __name__ == "__main__":
    validar_modelo()
