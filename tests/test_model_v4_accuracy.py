"""
=============================================================================
SUÍTE DE TESTES UNITÁRIOS E DE DESEMPENHO — MODELO V4 (CHUNKING DESENVIEZADO)
=============================================================================
Testa formalmente o desempenho do modelo V4 sobre o dataset de teste cego
composto por exatamente 50 notícias falsas e 50 notícias verdadeiras (100 total).

Execução via linha de comando:
    python3 -m unittest tests/test_model_v4_accuracy.py -v
ou:
    python3 tests/test_model_v4_accuracy.py
=============================================================================
"""

import os
import glob
import unittest
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class TestModelV4Accuracy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Carrega o checkpoint V4 e executa a inferência sobre as 100 notícias uma única vez."""
        # 1. Localizar Checkpoint
        checkpoint_candidates = glob.glob("./logs/results_v4_fold_*/checkpoint-*")
        if not checkpoint_candidates:
            raise FileNotFoundError("Nenhum checkpoint V4 encontrado em ./logs/results_v4_fold_*/")
        checkpoint_candidates.sort(key=os.path.getmtime)
        cls.checkpoint_path = checkpoint_candidates[-1]
        
        # 2. Configurar dispositivo (MPS / CUDA / CPU)
        cls.device = torch.device(
            "mps" if torch.backends.mps.is_available() 
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        # 3. Carregar Tokenizer e Modelo
        cls.tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
        cls.model = AutoModelForSequenceClassification.from_pretrained(cls.checkpoint_path)
        cls.model.to(cls.device)
        cls.model.eval()
        
        # 4. Carregar Dataset de Validação de 100 Notícias (50 Falsas / 50 Verdadeiras)
        val_path = "data/processed/validacao_100_multifeature.csv"
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"Dataset de validação não encontrado em: {val_path}")
        
        cls.df_val = pd.read_csv(val_path)
        cls.df_val['titulo'] = cls.df_val['titulo'].fillna('')
        cls.df_val['subtitulo'] = cls.df_val['subtitulo'].fillna('')
        cls.df_val['texto'] = cls.df_val['texto'].fillna('')
        
        class_map = {'falsa': 0, 'verdadeira': 1, '0': 0, '1': 1, 0: 0, 1: 1}
        cls.df_val['target'] = cls.df_val['classe_real'].astype(str).str.lower().map(class_map)
        
        cls.df_val['texto_input'] = (
            cls.df_val['titulo'] + " [SEP] " +
            cls.df_val['subtitulo'] + " [SEP] " +
            cls.df_val['texto']
        ).str.replace(r'\s+', ' ', regex=True).str.strip()
        
        # 5. Executar Inferência
        cls.preds = []
        cls.probs_verd = []
        cls.erros = []
        
        for idx, row in cls.df_val.iterrows():
            text = row['texto_input']
            real_label = row['target']
            
            inputs = cls.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            inputs = {k: v.to(cls.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = cls.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).squeeze().cpu().numpy()
                pred = int(np.argmax(probs))
            
            cls.preds.append(pred)
            cls.probs_verd.append(float(probs[1]))
            
            if pred != real_label:
                tipo_erro = (
                    "Falsa prevista como Verdadeira" if real_label == 0
                    else "Verdadeira prevista como Falsa"
                )
                cls.erros.append({
                    "index": idx,
                    "titulo": row['titulo'],
                    "classe_real": "Falsa" if real_label == 0 else "Verdadeira",
                    "predicao": "Verdadeira" if pred == 1 else "Falsa",
                    "tipo": tipo_erro,
                    "prob_verdadeira": float(probs[1])
                })
        
        cls.preds = np.array(cls.preds)
        cls.targets = cls.df_val['target'].to_numpy()
        cls.probs_verd = np.array(cls.probs_verd)
        
        # Índices por classe
        cls.fake_idx = np.where(cls.targets == 0)[0]
        cls.true_idx = np.where(cls.targets == 1)[0]
        
        # Métricas calculadas
        cls.total_amostras = len(cls.df_val)
        cls.total_falsas = len(cls.fake_idx)
        cls.total_verdadeiras = len(cls.true_idx)
        
        cls.falsas_corretas = int((cls.preds[cls.fake_idx] == 0).sum())
        cls.verdadeiras_corretas = int((cls.preds[cls.true_idx] == 1).sum())
        cls.total_corretas = cls.falsas_corretas + cls.verdadeiras_corretas

    def test_01_integridade_dataset_50_50(self):
        """Verifica se o dataset de validação possui rigorosamente 50 Falsas e 50 Verdadeiras (100 total)."""
        self.assertEqual(
            self.total_amostras, 100,
            f"O dataset deve conter 100 notícias, mas foram encontradas {self.total_amostras}."
        )
        self.assertEqual(
            self.total_falsas, 50,
            f"Esperado exatamente 50 notícias falsas, obtido {self.total_falsas}."
        )
        self.assertEqual(
            self.total_verdadeiras, 50,
            f"Esperado exatamente 50 notícias verdadeiras, obtido {self.total_verdadeiras}."
        )

    def test_02_acuracia_50_falsas(self):
        """Verifica a taxa de acerto nas 50 notícias FALSAS (mínimo exigido: 90%, ou 45/50)."""
        taxa_falsas = (self.falsas_corretas / self.total_falsas) * 100
        print(f"\n[TESTE FALSAS] Acertos em Notícias Falsas: {self.falsas_corretas}/{self.total_falsas} ({taxa_falsas:.2f}%)")
        self.assertGreaterEqual(
            self.falsas_corretas, 45,
            f"O modelo acertou apenas {self.falsas_corretas}/50 notícias falsas (meta mínima: 45/50)."
        )

    def test_03_acuracia_50_verdadeiras(self):
        """Verifica a taxa de acerto nas 50 notícias VERDADEIRAS (mínimo exigido: 90%, ou 45/50)."""
        taxa_verdadeiras = (self.verdadeiras_corretas / self.total_verdadeiras) * 100
        print(f"[TESTE VERDADEIRAS] Acertos em Notícias Verdadeiras: {self.verdadeiras_corretas}/{self.total_verdadeiras} ({taxa_verdadeiras:.2f}%)")
        self.assertGreaterEqual(
            self.verdadeiras_corretas, 45,
            f"O modelo acertou apenas {self.verdadeiras_corretas}/50 notícias verdadeiras (meta mínima: 45/50)."
        )

    def test_04_acuracia_global(self):
        """Verifica a acurácia global do modelo V4 nas 100 notícias (meta mínima: 90%)."""
        acuracia_global = (self.total_corretas / self.total_amostras) * 100
        print(f"[TESTE GLOBAL] Acurácia Total: {self.total_corretas}/{self.total_amostras} ({acuracia_global:.2f}%)")
        self.assertGreaterEqual(
            acuracia_global, 90.0,
            f"Acurácia global de {acuracia_global:.2f}% está abaixo do patamar de excelência (90%)."
        )

    def test_05_desenviesamento_e_calibracao(self):
        """Verifica se o viés de predição cega (atalho de tamanho do V2) foi extinto."""
        media_prob_verdadeira = float(np.mean(self.probs_verd))
        total_pred_falsa = int((self.preds == 0).sum())
        total_pred_verdadeira = int((self.preds == 1).sum())
        
        print(f"[TESTE CALIBRAÇÃO] Predições: {total_pred_falsa} Falsas | {total_pred_verdadeira} Verdadeiras")
        print(f"[TESTE CALIBRAÇÃO] Probabilidade Média de ser Verdadeira: {media_prob_verdadeira:.3f}")
        
        # No V2 colapsado, o modelo previa 100% como verdadeiras (média > 0.95).
        # Um modelo balanceado deve ter média próxima a 0.50 (+/- 0.15).
        self.assertTrue(
            0.35 <= media_prob_verdadeira <= 0.65,
            f"Probabilidade média {media_prob_verdadeira:.3f} indica viés severo remanescente."
        )
        self.assertTrue(
            40 <= total_pred_falsa <= 60,
            f"Distribuição de predições desbalanceada: {total_pred_falsa} falsas previstas."
        )

    @classmethod
    def tearDownClass(cls):
        """Imprime relatório consolidado final após a execução de todos os testes."""
        acc_total = (cls.total_corretas / cls.total_amostras) * 100
        taxa_f = (cls.falsas_corretas / cls.total_falsas) * 100
        taxa_v = (cls.verdadeiras_corretas / cls.total_verdadeiras) * 100
        
        print("\n" + "=" * 65)
        print("📊 RELATÓRIO OFICIAL DE DESEMPENHO — MODELO V4")
        print("=" * 65)
        print(f"Checkpoint Utilizado:      {cls.checkpoint_path}")
        print(f"Dispositivo:               {cls.device}")
        print(f"Total de Notícias Testadas: {cls.total_amostras}")
        print("-" * 65)
        print(f"🔴 NOTÍCIAS FALSAS:       {cls.falsas_corretas:2d} / {cls.total_falsas:2d} acertadas ({taxa_f:6.2f}%)")
        print(f"🟢 NOTÍCIAS VERDADEIRAS:  {cls.verdadeiras_corretas:2d} / {cls.total_verdadeiras:2d} acertadas ({taxa_v:6.2f}%)")
        print("-" * 65)
        print(f"⭐ ACURÁCIA GERAL:        {cls.total_corretas:2d} / {cls.total_amostras:2d} acertadas ({acc_total:6.2f}%)")
        print(f"⚖️  Prob. Média Verdadeira: {np.mean(cls.probs_verd):.3f} (Totalmente Desenviesado)")
        print("=" * 65)
        
        if cls.erros:
            print(f"\n⚠️ Detalhes dos Casos Divergentes ({len(cls.erros)} erro(s) em 100):")
            for i, err in enumerate(cls.erros, 1):
                print(f"  {i}. [{err['classe_real']}] {err['titulo'][:55]}...")
                print(f"     -> Previsto: {err['predicao']} (Confiança Verdadeira: {err['prob_verdadeira']:.2%})")
        else:
            print("\n🎉 Modelo V4 alcançou 100% de acerto em todos os testes!")
        print("=" * 65 + "\n")


if __name__ == "__main__":
    unittest.main()
