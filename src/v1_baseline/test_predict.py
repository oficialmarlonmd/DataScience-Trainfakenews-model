import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def predict_news(text, model_dir="./bert_fakenews_model"):
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Carregando modelo de: {model_dir} (Dispositivo: {device.upper()})...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    inputs = tokenizer(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        pred_label = int(torch.argmax(outputs.logits, dim=-1).cpu().numpy()[0])

    classes = ['FALSA', 'VERDADEIRA']
    print("\n" + "="*50)
    print(f"📰 Texto analisado: {text[:150]}..." if len(text) > 150 else f"📰 Texto analisado: {text}")
    print("="*50)
    print(f"🎯 Resultado: Notícia {classes[pred_label]}")
    print(f"   • Probabilidade Falsa:      {probs[0]*100:.2f}%")
    print(f"   • Probabilidade Verdadeira: {probs[1]*100:.2f}%")
    print("="*50 + "\n")
    return classes[pred_label], probs

if __name__ == "__main__":
    if len(sys.argv) > 1:
        sample_text = " ".join(sys.argv[1:])
    else:
        sample_text = "Ministério da Saúde inicia nova campanha de vacinação contra a gripe em todo o país."
    
    predict_news(sample_text)
