# 🧠 Classificação de Câncer de Mama com Redes Neurais

Projeto de Machine Learning com redes neurais artificiais para classificar tumores como **Benignos** ou **Malignos** usando o dataset Wisconsin Breast Cancer.

## 📋 Descrição

- **Objetivo:** Classificação binária (Benigno / Maligno)
- **Dataset:** [Wisconsin Breast Cancer](https://github.com/stedy/Machine-Learning-with-R-datasets/blob/master/wisc_bc_data.csv)
- **Ferramenta:** Python + TensorFlow/Keras

## 🏗️ Arquitetura da Rede Neural

```
Input (30 features)
    ↓
Dense(64, ReLU) + Dropout(0.3)
    ↓
Dense(32, ReLU) + Dropout(0.3)   ← Camada oculta 1
    ↓
Dense(16, ReLU)                  ← Camada oculta 2
    ↓
Dense(1, Sigmoid)                ← Saída
```

## 🔄 Etapas do Projeto

1. **EDA** — Análise exploratória com gráficos de distribuição, correlação e boxplot
2. **Pré-processamento** — Remoção de ID, encoding do target, normalização (StandardScaler), split 80/20
3. **Treinamento** — Rede neural com 2 camadas ocultas, EarlyStopping, 150 épocas
4. **Avaliação** — Acurácia, F1-Score, Precision, Recall, Matriz de Confusão

## 🚀 Como executar

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn
python neural_network_cancer.py
```

## 📊 Resultados esperados

- Acurácia no teste: ~96–98%
- Gráficos gerados:
  - `eda_distribuicao_target.png`
  - `eda_correlacao.png`
  - `eda_boxplot.png`
  - `treinamento_curvas.png`
  - `matriz_confusao.png`
