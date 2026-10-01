"""
=================================================================
MONITOR INTELIGENTE DO PIPELINE (V3 Staging -> Validação -> V4 Chunking)
1. Monitora o processo em execução do V3 (train_v3_recalibrado.py).
2. Assim que o V3 concluir, roda automaticamente a validação nas 84 notícias.
3. Inicia o treinamento do modelo V4 (Chunking / Desenviesado).
4. Ao final, roda a validação do V4 e exibe a comparação definitiva.
=================================================================
"""
import time
import subprocess
import sys
import os

def is_v3_running():
    try:
        output = subprocess.check_output(["pgrep", "-f", "train_v3_recalibrado.py"]).decode()
        return len(output.strip()) > 0
    except subprocess.CalledProcessError:
        return False

print("👀 Monitor do Pipeline Iniciado!")
print("Acompanhando o modelo V3 em execução...")

inicio_espera = time.time()
while is_v3_running():
    minutos_decorridos = (time.time() - inicio_espera) / 60
    print(f"⏳ V3 ainda em treinamento... ({minutos_decorridos:.1f} min monitorando) - Checando novamente em 30s")
    time.sleep(30)

print("\n" + "=" * 60)
print("🎉 O TREINAMENTO DO MODELO V3 TERMINOU!")
print("=" * 60)

# 1. Executar validação do V3
print("\n📊 [PASSO 1/2] Executando Validação Automática do V3 nas notícias reais...")
subprocess.run([sys.executable, "src/validar_v3.py"])

# 2. Executar treinamento do V4
print("\n🚀 [PASSO 2/2] Iniciando Treinamento V4 (Dataset Chunked / Desenviesado)...")
ret = subprocess.run([sys.executable, "src/train_v4_chunking.py"])

if ret.returncode == 0:
    print("\n✅ Treinamento V4 concluído com sucesso!")
    print("\n📊 Executando Validação do V4...")
    subprocess.run([sys.executable, "src/validar_v4.py"])
else:
    print("❌ Ocorreu um erro durante o treinamento do V4.")
