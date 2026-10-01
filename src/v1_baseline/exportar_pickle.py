import pickle
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import warnings

# Ignorar warnings
warnings.filterwarnings('ignore')

print("⏳ Carregando o modelo e o tokenizador salvos na pasta 'bert_fakenews_model'...")
model_dir = './bert_fakenews_model'

model = AutoModelForSequenceClassification.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# A forma mais robusta de exportar um modelo transformer para consumo genérico
# via pickle é envelopá-lo em um Pipeline do Hugging Face.
# Assim, o .pkl já levará junto o tokenizador e fará o pré-processamento automaticamente.
print("🔧 Envelopando em um pipeline de classificação...")
nlp_pipeline = pipeline(
    "text-classification", 
    model=model, 
    tokenizer=tokenizer,
    device=-1 # Forçar uso de CPU no carregamento inicial para máxima compatibilidade entre servidores
)

pickle_path = 'modelo_fakenews.pkl'
print(f"📦 Exportando para {pickle_path}...")

with open(pickle_path, 'wb') as f:
    pickle.dump(nlp_pipeline, f)

print("✅ Pickle exportado com sucesso!")
print("\n📝 PARA CONSUMIR ESTE ARQUIVO DEPOIS, BASTA FAZER:")
print("import pickle")
print("with open('modelo_fakenews.pkl', 'rb') as f:")
print("    modelo_carregado = pickle.load(f)")
print("resultado = modelo_carregado('Texto da noticia aqui')")
print("print(resultado)")
